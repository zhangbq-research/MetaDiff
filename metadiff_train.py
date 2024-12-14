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

# from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.resnet12_2 import resnet12
# from models.protonet_metanet1 import ProtonetMetaLearner
# from models.classifier import LinearClassifier, NNClassifier, distLinear
# from models.PredTrainHead import LinearRotateHead, DCLHead, DistRotateHead
from models.diffusion import get_diffusion_from_args

from utils import set_gpu, Timer, count_accuracy, check_dir, log

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


    # Choose the classification head

    cls_head = get_diffusion_from_args().cuda()
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速

def train(opt, dataset_train, dataset_val, data_loader):
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 10000,  # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.save_path, 'best_pretrain_model_resnet.pth'))
    # saved_models = torch.load(os.path.join(opt.save_path, 'best_model2.pth'))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()

    import pickle
    with open(os.path.join(opt.save_path, "mini_imagenet_optimal_prototype.pickle"), 'rb') as handle:
        prototype_ground_true = pickle.load(handle)
    prototype_ground_true_dict = prototype_ground_true['dict']
    prototype_ground_true_array = prototype_ground_true['array'].cuda()

    optimizer = torch.optim.Adam(cls_head.parameters(), lr=0.0001, weight_decay=5e-4)

    lambda_epoch = lambda e: 1.0 if e < 100000 else (0.1 if e < 30 else 0.01 if e < 40 else (0.001))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
            epoch, epoch_learning_rate))

        _ = [x.train() for x in (cls_head, )]
        # _, _ = [x.train() for x in (cls_head, embedding_net)]

        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, all_k, _ = [x.cuda() for x in batch]

            # prototype_ground_true = torch.gather(prototype_ground_true_array, 0, all_k)
            prototype_ground_true = prototype_ground_true_array[all_k, :]

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            with torch.no_grad():
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
                # emb_support = F.normalize(emb_support, dim=-1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
                # emb_query = F.normalize(emb_query, dim=-1)

            loss = cls_head(prototype_ground_true, emb_query, emb_support, labels_support, labels_query, opt.train_way, opt.train_shot)



            if (i % 1000 == 0):
                logit_query = cls_head.sample(emb_query, emb_support, labels_support, labels_query, opt.train_way,
                                              opt.train_shot)
                # loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
                acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

                train_accuracies.append(acc.item())
                train_losses.append(loss.item())
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {}\tAccuracy: {} % ({} %)'.format(
                    epoch, i, loss.item(), train_acc_avg, acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (cls_head, embedding_net)]

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query
            with torch.no_grad():
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)
                # emb_support = F.normalize(emb_support, dim=-1)
                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)
                # emb_query = F.normalize(emb_query, dim=-1)

            logit_query = cls_head.sample(emb_query, emb_support, labels_support, labels_query, opt.test_way, opt.val_shot)
            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            # torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}, \
            #            os.path.join(opt.save_path, 'best_model_val_double_dir_opt_cub_{}_shot.pth'.format(opt.val_shot)))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))


def test(opt, n_iter, dataset_train, dataset_val, dataset_test, data_loader):
    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.save_path, 'best_model_val_double_dir_opt_cub_5_shot.pth'))
    # saved_models = torch.load(os.path.join(opt.save_path, 'best_model2.pth'))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()

    max_val_acc = 0.0
    max_test_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    # Evaluate on the validation split
    _, _ = [x.eval() for x in (cls_head, embedding_net)]

    test_accuracies = []
    test_losses = []
    epoch = 9
    for i, batch in enumerate(tqdm(dloader_test(epoch)), 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

        test_n_support = opt.test_way * opt.val_shot
        test_n_query = opt.test_way * opt.val_query

        with torch.no_grad():
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            # emb_support = F.normalize(emb_support, dim=-1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)
            # emb_query = F.normalize(emb_query, dim=-1)

        logit_querys = cls_head(emb_query, emb_support, labels_support, labels_query, opt.test_way, opt.val_shot, is_train=False, update_step_test=n_iter)
        # logit_querys = logit_querys[-1:]
        for kk, logit_query in enumerate(logit_querys):
            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

        test_accuracies.append(acc.item())
        test_losses.append(loss.item())

    test_acc_avg = np.mean(np.array(test_accuracies))
    test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.val_episode)

    test_loss_avg = np.mean(np.array(test_losses))

    if test_acc_avg > max_test_acc:
        max_test_acc = test_acc_avg
        log(log_file_path, 'Test Loss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
            .format(test_loss_avg, test_acc_avg, test_acc_ci95))
    else:
        log(log_file_path, 'Test Loss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
            .format(test_loss_avg, test_acc_avg, test_acc_ci95))

if __name__ == '__main__':
    seed_torch(21)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=1000,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=5,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
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
    parser.add_argument('--pre_head', type=str, default='LinearRotateNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')


    opt = parser.parse_args()
    
    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    train(opt, dataset_train, dataset_val, data_loader)