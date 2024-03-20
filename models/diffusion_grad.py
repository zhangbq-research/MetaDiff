import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

import argparse
import torchvision
from .attention import SelfAttention, CrossAttention
# from models.unet import UNet

# from .ema import EMA
# from .utils import extract

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=320):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=512):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))  # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)  # (n, c, hw)
        x = x.view((n, c, h, w))  # (n, c, h, w)

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(ResidualBlock(512, 256), AttentionBlock(8, 32)),
            SwitchSequential(ResidualBlock(256, 128), AttentionBlock(8, 16)),
        ])
        self.bottleneck = SwitchSequential(
            ResidualBlock(128, 128),
            AttentionBlock(8, 16),
            ResidualBlock(128, 128),
        )
        self.decoders = nn.ModuleList([
            SwitchSequential(ResidualBlock(128+128, 256), AttentionBlock(8, 32)),
            SwitchSequential(ResidualBlock(256+256, 512), AttentionBlock(8, 64)),
        ])

        self.pos_emb = nn.Linear(5, 512, bias=False)

    def get_time_embedding(self, timestep):
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160).cuda()
        x = timestep[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def forward(self, x, time = None, emb_query = None, emb_support = None, labels_support = None, labels_query = None, n_way = None, k_shot = None):
        time = self.get_time_embedding(time)

        tasks_per_batch = emb_query.size(0)
        n_support = emb_support.size(1)


        support_labels_one_hot = one_hot(labels_support.reshape(-1), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

        pos_emb = self.pos_emb(support_labels_one_hot)
        context = emb_support + pos_emb

        prototype_labels = torch.LongTensor(list(range(n_way))).cuda()
        prototype_labels = one_hot(prototype_labels, n_way).view(1, n_way, n_way)
        prototype_pos_emb = self.pos_emb(prototype_labels)

        x = x + prototype_pos_emb.permute(0, 2, 1).unsqueeze(-1).repeat(tasks_per_batch, 1, 1, 1)


        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class EMA():
    def __init__(self, decay):
        self.decay = decay

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

class DMFunc(nn.Module):

    def __init__(self):
        super(DMFunc, self).__init__()
        self.nfe = 0
        self.scale_factor = 10
        self.time_scale = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Linear(160, 1),
            nn.Sigmoid()
        )
        # self.res = nn.Sequential(
        #     nn.Linear(512+320, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        # )

        self.time_emb1 = nn.Linear(320, 256)
        self.proto_inf1 = nn.Linear(512, 256)

        self.time_emb2 = nn.Linear(320, 128)
        self.proto_inf2 = nn.Linear(256, 128)

        self.time_emb3 = nn.Linear(320, 128)
        self.proto_inf3 = nn.Linear(128, 128)

        self.time_emb4 = nn.Linear(320, 256)
        self.proto_inf4 = nn.Linear(128, 256)

        self.time_emb5 = nn.Linear(320, 512)
        self.proto_inf5 = nn.Linear(256, 512)

        # self.time_emb1 = nn.Linear(320, 256//4)
        # self.proto_inf1 = nn.Linear(512//4, 256//4)
        #
        # self.time_emb2 = nn.Linear(320, 128//4)
        # self.proto_inf2 = nn.Linear(256//4, 128//4)
        #
        # self.time_emb3 = nn.Linear(320, 128//4)
        # self.proto_inf3 = nn.Linear(128//4, 128//4)
        #
        # self.time_emb4 = nn.Linear(320, 256//4)
        # self.proto_inf4 = nn.Linear(128//4, 256//4)
        #
        # self.time_emb5 = nn.Linear(320, 512//4)
        # self.proto_inf5 = nn.Linear(256//4, 512//4)

    def get_time_embedding(self, timestep):
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160).cuda()
        x = timestep[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def forward(self, x, time, emb_query, emb_support, labels_support, labels_query, n_way, k_shot):
        tasks_per_batch = emb_support.shape[0]
        n_support = emb_support.shape[1]
        support_labels = one_hot(labels_support.view(tasks_per_batch * n_support), n_way)
        support_labels = support_labels.view(tasks_per_batch, n_support, -1)

        # encoding time
        time = self.get_time_embedding(time)
        scale = self.time_scale(time)

        nb, nd, nw, _ = x.shape
        prototypes = x.detach()[:, :, :, 0].permute(0, 2, 1)

        #
        weights = prototypes
        weights.requires_grad = True

        # logits = 10 * F.cosine_similarity(
        #     emb_support.unsqueeze(2).expand(-1, -1, n_way, -1),
        #     weights.unsqueeze(1).expand(-1, emb_support.shape[1], -1, -1),
        #     dim=-1) + torch.sum(weights*weights) * 5e-4
        # loss = nn.MSELoss()(F.softmax(logits.reshape(-1, n_way), dim=1), support_labels.reshape(-1, n_way))
        # loss = nn.CrossEntropyLoss()(logits.reshape(-1, n_way), labels_support.reshape(-1))

        diff = weights.unsqueeze(1).expand(-1, emb_support.shape[1], -1, -1) - emb_support.unsqueeze(2).expand(-1, -1, n_way, -1)
        loss = torch.sum(torch.sum(diff*diff, dim=-1) * support_labels)

        # logits = F.softmax(-1 * torch.sum(diff*diff, dim=-1) / 512, dim=-1)
        # # loss = nn.MSELoss()(F.softmax(logits.reshape(-1, n_way), dim=1), support_labels.reshape(-1, n_way))
        # loss = nn.CrossEntropyLoss()(logits.reshape(-1, n_way), labels_support.reshape(-1))

        # compute grad and update inner loop weights
        grads = torch.autograd.grad(loss, weights)
        x = grads[0].detach()

        # # prototypes_ = (prototypes+prototypes_fs)/2
        # prototypes_ = prototypes

        # query_labels = torch.nn.functional.cosine_similarity(
        #     emb_query.unsqueeze(2).expand(-1, -1, prototypes_.shape[1], -1),
        #     prototypes_.unsqueeze(1).expand(-1, emb_query.shape[1], -1, -1),
        #     dim=-1)
        # query_labels = query_labels * self.scale_factor
        # query_labels = F.softmax(query_labels, dim=-1)
        # data_labels = torch.cat([support_labels, query_labels], dim=1).unsqueeze(dim=-1).permute(0, 2, 1, 3)

        # topk, indices = torch.topk(data_labels, k_shot, dim=2)
        # mask = torch.zeros_like(data_labels)
        # mask = mask.scatter(2, indices, 1)
        # data_labels = data_labels * mask

        # data_labels = support_labels.unsqueeze(dim=-1).permute(0, 2, 1, 3)
        #
        # diff_weights = data_labels / torch.sum(data_labels, dim=2, keepdim=True)
        #
        # # cal vector fild
        # # all_x = torch.cat([emb_support, emb_query], dim=1)
        # all_x = emb_support
        # x = prototypes
        # x_left = x.unsqueeze(dim=2).expand(-1, -1, all_x.shape[1], -1)
        # all_x_right = all_x.unsqueeze(dim=1).expand(-1, n_way, -1, -1)
        # diff = (x_left - all_x_right)
        # diff = scale.unsqueeze(dim=-1) * torch.sum((diff_weights * diff), dim=2) + (1-scale.unsqueeze(dim=-1))*self.res(torch.cat([prototypes, time.unsqueeze(dim=1).repeat(1,5,1)], dim=-1))

        # x = torch.sum((diff_weights * diff), dim=2)

        time_ = self.time_emb1(time).unsqueeze(dim=1).repeat(1,5,1)
        x_1 = self.proto_inf1(x)
        x_1 = F.softplus(x_1 * time_)

        time_ = self.time_emb2(time).unsqueeze(dim=1).repeat(1,5,1)
        x_2 = self.proto_inf2(x_1)
        x_2 = F.softplus(x_2 * time_)

        time_ = self.time_emb3(time).unsqueeze(dim=1).repeat(1,5,1)
        x_3 = self.proto_inf3(x_2)
        x_3 = F.softplus(x_3 * time_) + x_2

        time_ = self.time_emb4(time).unsqueeze(dim=1).repeat(1,5,1)
        x_4 = self.proto_inf4(x_3)
        x_4 = F.softplus(x_4 * time_) + x_1

        time_ = self.time_emb5(time).unsqueeze(dim=1).repeat(1,5,1)
        x_5 = self.proto_inf5(x_4)
        x_5 = x_5 * time_ + x

        # diff = scale.unsqueeze(dim=-1) * torch.sum((diff_weights * diff), dim=2) + (
        #             1 - scale.unsqueeze(dim=-1)) * x

        # diff = x_5 * scale.unsqueeze(dim=-1)

        diff = scale.unsqueeze(dim=-1) * x + (
                    1 - scale.unsqueeze(dim=-1)) * x_5

        return diff.permute(0, 2, 1).unsqueeze(-1)


class DMFunc1(nn.Module):

    def __init__(self):
        super(DMFunc1, self).__init__()
        self.nfe = 0
        self.scale_factor = 10

        self.time_emb1 = nn.Linear(320, 256)
        self.proto_inf1 = nn.Linear(512, 256)
        self.spt_inf1 = nn.Linear(512, 256)

        self.time_emb2 = nn.Linear(320, 128)
        self.proto_inf2 = nn.Linear(256, 128)
        self.spt_inf2 = nn.Linear(256, 128)

        self.time_emb3 = nn.Linear(320, 128)
        self.proto_inf3 = nn.Linear(128, 128)
        self.spt_inf3 = nn.Linear(128, 128)

        self.time_emb4 = nn.Linear(320, 256)
        self.proto_inf4 = nn.Linear(128, 256)
        self.spt_inf4 = nn.Linear(128, 256)

        self.time_emb5 = nn.Linear(320, 512)
        self.proto_inf5 = nn.Linear(256, 512)
        self.spt_inf5 = nn.Linear(256, 512)

    def get_time_embedding(self, timestep):
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160).cuda()
        x = timestep[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def forward(self, x, time, emb_query, emb_support, labels_support, labels_query, n_way, k_shot):
        tasks_per_batch = emb_support.shape[0]
        n_support = emb_support.shape[1]

        # encoding time
        time = self.get_time_embedding(time)

        nb, nd, nw, _ = x.shape
        x = x.permute(0, 2, 3, 1)

        sorted_emb_support = []
        for way_i in range(n_way):
            temp = []
            for bi in range(tasks_per_batch):
                temp.append(emb_support[bi, labels_support[bi, :]==way_i, :])
            temp = torch.stack(temp, dim=0)
            sorted_emb_support.append(temp)
        sorted_emb_support = torch.stack(sorted_emb_support, dim=1)

        time_ = self.time_emb1(time).unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.proto_inf1(x)
        spt_emb = self.spt_inf1(sorted_emb_support)
        x = F.softplus(x * time_ * spt_emb)
        x_1 = x
        time_ = self.time_emb2(time).unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.proto_inf2(x)
        spt_emb = self.spt_inf2(spt_emb)
        x = F.softplus(x * time_ * spt_emb)
        x_2 = x
        time_ = self.time_emb3(time).unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.proto_inf3(x)
        spt_emb = self.spt_inf3(spt_emb)
        x = F.softplus(x * time_ * spt_emb) + x_2

        time_ = self.time_emb4(time).unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.proto_inf4(x)
        spt_emb = self.spt_inf4(spt_emb)
        x = F.softplus(x * time_ * spt_emb) + x_1
        time_ = self.time_emb5(time).unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.proto_inf5(x)
        spt_emb = self.spt_inf5(spt_emb)
        x = F.softplus(x * time_ * spt_emb) + x

        x = x.mean(dim=-2)
        return x.permute(0, 2, 1).unsqueeze(-1)


class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """

    def __init__(
            self,
            model,
            img_size,
            img_channels,
            num_classes,
            betas,
            loss_type="l2",
            ema_decay=0.9999,
            ema_start=5000,
            ema_update_rate=1,
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    # @torch.no_grad()
    def remove_noise(self, x, t, emb_query, emb_support, labels_support, labels_query, n_way, k_shot, use_ema=True):
        if use_ema:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, emb_query, emb_support, labels_support, labels_query, n_way, k_shot)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, emb_query, emb_support, labels_support, labels_query, n_way, k_shot)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    #@torch.no_grad()
    def sample(self, emb_query, emb_support, labels_support, labels_query, n_way, k_shot, use_ema=True):
        batch_size = emb_query.shape[0]
        device = emb_query.device
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, emb_query, emb_support, labels_support, labels_query, n_way, k_shot, use_ema)

            # if t > 0:
            #     x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            # if t > 0:
            #     x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        prototype = x.detach()[:, :, :, 0].permute(0, 2, 1)
        logits = 10 * F.cosine_similarity(
            emb_query.unsqueeze(2).expand(-1, -1, n_way, -1),
            prototype.unsqueeze(1).expand(-1, emb_query.shape[1], -1, -1),
            dim=-1)
        # logits = torch.sum(emb_query.unsqueeze(2).expand(-1, -1, n_way, -1)*
        #     prototype.unsqueeze(1).expand(-1, emb_query.shape[1], -1, -1), dim=-1)
        return logits

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, emb_query, emb_support, labels_support, labels_query, n_way, k_shot):
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, emb_query, emb_support, labels_support, labels_query, n_way, k_shot)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        # loss = F.l1_loss(estimated_noise, noise)

        return loss

    def forward(self, x, emb_query, emb_support, labels_support, labels_query, n_way, k_shot):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        b, c, h, w = x.shape
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, emb_query, emb_support, labels_support, labels_query, n_way, k_shot)


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def get_diffusion_from_args():
    num_timesteps = 1000
    schedule = "linear"
    loss_type = "l2"
    use_labels = False

    base_channels = 128
    channel_mults = (1, 2, 2, 2)
    num_res_blocks = 2
    time_emb_dim = 128 * 4
    norm = "gn"
    dropout = 0.1
    activation = "silu"
    attention_resolutions = (1,)

    ema_decay = 0.9999
    ema_update_rate = 1
    schedule_low = 1e-4
    schedule_high = 0.02

    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = DMFunc()

    if schedule == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (5, 1), 512, 10,
        betas,
        ema_decay=ema_decay,
        ema_update_rate=ema_update_rate,
        ema_start=2000,
        loss_type=loss_type,
    )
    return diffusion