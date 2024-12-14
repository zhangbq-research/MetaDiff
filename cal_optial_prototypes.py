# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.resnet12_2 import resnet12
from utils import set_gpu, Timer, count_accuracy, check_dir, log
import pickle

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
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
        fea_dim = 64
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        network = resnet12().cuda()
        network = torch.nn.DataParallel(network)
        fea_dim = 512
    else:
        print ("Cannot recognize the network type")
        assert(False)
    # info_max_layer = InfoMaxLayer().cuda()
    return network

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, dataset_test, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=100,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=20,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=600,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='CosineNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--pre_head', type=str, default='add_margin',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')

    opt = parser.parse_args()
    
    (dataset_train, dataset_val, dataset_test, data_loader) = get_dataset(opt)

    data_loader_pre = torch.utils.data.DataLoader
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader_pre(
        dataset=dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    embedding_net = get_model(opt)

    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.save_path, 'best_pretrain_model_resnet.pth'))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()

    embs = []
    labels = []
    for i, batch in enumerate(tqdm(dloader_train), 1):
        data, label = [x.cuda() for x in batch]
        with torch.no_grad():
            emb = embedding_net(data)
        embs.append(emb)
        labels.append(label)
    embs = torch.cat(embs, dim=0).cpu()
    labels = list(torch.cat(labels, dim=0).cpu().numpy())
    label2index = {}
    for k, v in enumerate(labels):
        v = int(v)
        if v not in label2index.keys():
            label2index[v] = []
            label2index[v].append(k)
        else:
            label2index[v].append(k)

    label2optimal_proto = {}
    optimal_prototype = torch.zeros(64, 512).type_as(embs)
    for k, v in label2index.items():
        sub_embs = embs[v, :]
        label2optimal_proto[k] = torch.mean(sub_embs, dim=0).unsqueeze(dim=0)
        optimal_prototype[k] = torch.mean(sub_embs, dim=0)

    database = {'dict': label2optimal_proto, 'array': optimal_prototype}
    with open(os.path.join(opt.save_path, "mini_imagenet_optimal_prototype.pickle"), 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(opt.save_path, "mini_imagenet_optimal_prototype.pickle"), 'rb') as handle:
        part_prior = pickle.load(handle)
    print(1)