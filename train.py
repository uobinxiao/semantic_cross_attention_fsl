import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import numpy
from dataloader.mini_imagenet import MiniImageNet
from dataloader.cub import CUB
from dataloader.cifar100 import CIFAR100
from dataloader.tiered_imagenet import TieredImageNet
from dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from models.proxynet import ProxyNet
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from evaluation import evaluation
from tqdm import tqdm
import math
import torch.nn.functional as F
from torch.nn import init
import wandb
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr

def sharpen(p, T=0.2):
    '''Sharpening function as described in the paper.
    Increases confidence of the model in its predictions.
    Entropy minimization is implicitly achieved through this function.'''
    p_sharp = torch.pow(p, 1/T)/(torch.sum(torch.pow(p, 1/T), dim=0))
    
    return p_sharp

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

def train():
    wandb.init(project="multi_modal_fsl")
    os.environ["CUDA_VISIBLE_DEVICES"] = wandb.config.gpu_id

    save_best = wandb.config.save_best
    save_name = str(wandb.config.proxy_type) + "_" + str(wandb.config.classifier) + str(wandb.config.model_type) + "_" + str(wandb.config.dataset) + "_" + str(wandb.config.num_shot) + "_" + str(wandb.config.num_way) + ".pth"

    train_set = None
    val_set = None
    test_set = None
    image_size = 84
    if wandb.config.dataset == "MiniImageNet":
        train_set = MiniImageNet(setname = "train", image_size = image_size, if_augmentation = wandb.config.if_augmentation)
        val_set = MiniImageNet(setname = "val", image_size = image_size)
        test_set = MiniImageNet(setname = "test", image_size = image_size)

    if wandb.config.dataset == "CUB":
        train_set = CUB(setname = "train", image_size = image_size, if_augmentation = wandb.config.if_augmentation)
        val_set = CUB(setname = "val", image_size = image_size)
        test_set = CUB(setname = "test", image_size = image_size)

    if wandb.config.dataset == "TieredImageNet":
        train_set = TieredImageNet(setname = "train", image_size = image_size, if_augmentation = wandb.config.if_augmentation)
        val_set = TieredImageNet(setname = "val", image_size = image_size)
        test_set = TieredImageNet(setname = "test", image_size = image_size)

    if wandb.config.dataset == "CIFAR100":
        train_set = CIFAR100(setname = "train", image_size = image_size, if_augmentation = wandb.config.if_augmentation)
        val_set = CIFAR100(setname = "val", image_size = image_size)
        test_set = CIFAR100(setname = "test", image_size = image_size)

    train_sampler = CategoriesSampler(train_set.label, n_batch = wandb.config.num_train, n_cls = wandb.config.num_way, n_per = wandb.config.num_shot + wandb.config.num_query)
    train_loader = DataLoader(dataset=train_set, batch_sampler = train_sampler, num_workers= 8, pin_memory = True, worker_init_fn=worker_init_fn)
    val_sampler = CategoriesSampler(val_set.label, n_batch = wandb.config.num_val, n_cls = wandb.config.num_way, n_per = wandb.config.num_shot + wandb.config.num_query)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_sampler = CategoriesSampler(test_set.label, n_batch = wandb.config.num_test, n_cls = wandb.config.num_way, n_per = wandb.config.num_shot + wandb.config.num_query)
    test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    proxynet= ProxyNet(model_type = wandb.config.model_type, num_shot = wandb.config.num_shot, num_way = wandb.config.num_way, num_query = wandb.config.num_query, proxy_type = wandb.config.proxy_type, classifier = wandb.config.classifier).cuda()

    print(count_parameters(proxynet))

    optimizer = None
    if wandb.config.optimizer == "SGD":
        optimizer = torch.optim.SGD(proxynet.parameters(), lr = wandb.config.sgd_lr, weight_decay = 0.001)
    elif wandb.config.optimizer == "Adam":
        optimizer = torch.optim.Adam(proxynet.parameters())
    elif wandb.config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(proxynet.parameters())
    else:
        raise "Optimizer error"

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience = int(wandb.config.patience), factor = float(wandb.config.reduce_factor), min_lr = 0.00001)
    ce = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    label = torch.arange(wandb.config.num_way).repeat(wandb.config.num_query)
    #one hot encode
    one_hot_label = torch.zeros(len(label), label.max()+1).scatter_(1, label.unsqueeze(1), 1.).float().cuda()
    label = label.cuda()
    max_acc = 0
    max_test_acc = 0
    train_accuracy = 0
    min_loss = 0
    for epoch in range(wandb.config.num_epochs):
        total_rewards = 0
        total_loss = 0
        for i, batch in enumerate(train_loader, 1):
            proxynet.train()
            data = batch[0].cuda()
            text_label = batch[2]

            label_space = text_label[0: wandb.config.num_way]
            text_feature_list = []
            text_distance_list = []

            p = wandb.config.num_shot * wandb.config.num_way
            support, query = data[:p], data[p:]
            relation_score, _ = proxynet(support, query)

            loss = ce(-1 * relation_score, label)
            total_loss = total_loss + loss.item()
            _, predict_label = torch.min(relation_score, 1)
            rewards = [1 if predict_label[j]==label[j] else 0 for j in range(label.shape[0])]
            total_rewards += numpy.sum(rewards)
            
            proxynet.zero_grad()
            loss.backward()
            optimizer.step()

            episode = epoch * wandb.config.num_train + i + 1
            if episode % 100 == 0:
                print("episode:", epoch * wandb.config.num_train + i+1, "ce loss", total_loss / float(i + 1)) 
                train_accuracy = numpy.sum(total_rewards)/1.0 / wandb.config.num_query / wandb.config.num_way / wandb.config.num_train
                print('Train Accuracy of the model on the train :{:.2f} %'.format(100 * train_accuracy))
            if episode % 1000 == 0:
                acc, _ = evaluation(wandb.config, proxynet, val_loader, mode="val")
                if acc > max_acc:
                    max_acc = acc
                print("episode:", epoch * wandb.config.num_train + i+1,"val acc:", acc, " max val acc:", max_acc)
                wandb.log({"val_acc": acc})
                wandb.log({"max_val_acc": max_acc})

        scheduler.step(total_loss)
        print("sgd learning rate:", get_learning_rate(optimizer))

    #testing
    torch.save(proxynet.state_dict(), os.path.join("weights", save_name))
    test_acc, _, = evaluation(wandb.config, proxynet, test_loader, mode="test")
    wandb.log({"test_acc":test_acc})

if __name__ == "__main__":
    train()
