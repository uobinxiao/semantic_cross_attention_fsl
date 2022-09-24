import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import numpy
from dataloader.mini_imagenet import MiniImageNet
from dataloader.cub import CUB
from dataloader.tiered_imagenet import TieredImageNet
from dataloader.cifar100 import CIFAR100
from dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from models.multinet import MultiModalNet
from models.glove_text import GloveTextLabel
from models.fast_text import FastTextLabel
from models.bert_text import BertTextLabel
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.nn import CosineEmbeddingLoss
from evaluation import evaluation
from tqdm import tqdm
import math
import torch.nn.functional as F
from torch.nn import init
import wandb
import os
from timm.data.loader import create_loader

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
    save_name = "mmnet" + "_" + str(wandb.config.proxy_type) + "_" + str(wandb.config.classifier) + str(wandb.config.model_type) + "_" + str(wandb.config.dataset) + "_" + str(wandb.config.num_shot) + "_" + str(wandb.config.num_way) + "_"+ str(wandb.config.lam) + "_" + str(wandb.config.temperature) + "_" + str(wandb.config.scale) + ".pth"

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

    mmnet= MultiModalNet(wandb.config).cuda()

    optimizer = None
    if wandb.config.optimizer == "SGD":
        optimizer = torch.optim.SGD(mmnet.parameters(), lr = wandb.config.sgd_lr, weight_decay = 0.001)
    elif wandb.config.optimizer == "Adam":
        optimizer = torch.optim.Adam(mmnet.parameters())
    elif wandb.config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(mmnet.parameters())
    else:
        raise "Optimizer error"

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience = int(wandb.config.patience), factor = float(wandb.config.reduce_factor), min_lr = 0.00001)
    ce = nn.CrossEntropyLoss().cuda()
    mse = nn.MSELoss()
    kl = nn.KLDivLoss(reduction = "batchmean")
    cudnn.benchmark = True

    text_util = None
    if wandb.config.teacher_model == "Glove":
        text_util = GloveTextLabel(wandb.config.dataset, if_random = wandb.config.random_embedding)
    elif wandb.config.teacher_model == "Bert":
        text_util = BertTextLabel(wandb.config.dataset)
    else:
        text_util = FastTextLabel(wandb.config.dataset)

    label = torch.arange(wandb.config.num_way).repeat(wandb.config.num_query)
    #one hot encode
    one_hot_label = torch.zeros(len(label), label.max()+1).scatter_(1, label.unsqueeze(1), 1.).float().cuda()
    label = label.cuda()
    max_acc = 0
    max_test_acc = 0
    train_accuracy = 0
    for epoch in range(wandb.config.num_epochs):
        total_rewards = 0
        total_loss = 0
        total_ce_loss = 0
        total_aux_loss = 0
        aux_loss = 0
        for i, batch in enumerate(train_loader, 1):
            mmnet.train()
            data = batch[0].cuda()
            text_label = batch[2]

            if wandb.config.teacher_model == "Bert":
                text_feature_array = torch.zeros((wandb.config.num_way * (wandb.config.num_query + wandb.config.num_shot), 768))
            else:
                text_feature_array = torch.zeros((wandb.config.num_way * (wandb.config.num_query + wandb.config.num_shot), 300))

            for text_index, text_item in enumerate(text_label):
                text_feature = text_util.embed_text(text_item)
                text_feature_array[text_index, :] = torch.FloatTensor(text_feature)

            text_feature = torch.FloatTensor(text_feature_array).squeeze().cuda()

            p = wandb.config.num_shot * wandb.config.num_way
            support, query = data[:p], data[p:]
            relation_score, visual_feature, _, _ = mmnet(support, query)

            if wandb.config.aux_loss == "KL":
                aux_loss = kl(F.log_softmax(visual_feature / wandb.config.temperature, dim = 1), F.softmax(text_feature / wandb.config.temperature, dim = 1))
            elif wandb.config.aux_loss == "MSE":
                aux_loss = mse(visual_feature, text_feature)
            ce_loss = ce(-1 * relation_score, label)
            loss = (1 - wandb.config.lam) * ce_loss +  wandb.config.lam * aux_loss
            total_loss = total_loss + loss.item()
            total_ce_loss = total_ce_loss + ce_loss.item()
            total_aux_loss = total_aux_loss + aux_loss.item()
            _, predict_label = torch.min(relation_score, 1)
            rewards = [1 if predict_label[j]==label[j] else 0 for j in range(label.shape[0])]
            total_rewards += numpy.sum(rewards)
            
            mmnet.zero_grad()
            loss.backward()
            optimizer.step()

            episode = epoch * wandb.config.num_train + i + 1
            if episode % 100 == 0:
                print("episode:", epoch * wandb.config.num_train + i+1, "ce loss", total_ce_loss / float(i + 1), 
                        "kl loss", total_aux_loss / float(i + 1)) 
                train_accuracy = numpy.sum(total_rewards)/1.0 / wandb.config.num_query / wandb.config.num_way / wandb.config.num_train
                print('Train Accuracy of the model on the train :{:.2f} %'.format(100 * train_accuracy))
            if episode % 1000 == 0:
                acc, _ = evaluation(wandb.config, mmnet, val_loader, mode="val")
                #testing when val score increases
                if acc > max_acc:
                    max_acc = acc
                    max_test_acc, _ = evaluation(wandb.config, mmnet, test_loader, mode="test")
                    print("episode:", epoch * wandb.config.num_train + i+1,"test acc:", max_test_acc)
                    wandb.log({"test_acc": max_test_acc})
                print("episode:", epoch * wandb.config.num_train + i+1,"val acc:", acc, " max val acc:", max_acc, "test acc:", max_test_acc)
                wandb.log({"val_acc": acc})
                wandb.log({"max_val_acc": max_acc})

        scheduler.step(max_acc)
        print("sgd learning rate:", get_learning_rate(optimizer))

    #testing
    torch.save(mmnet.state_dict(), os.path.join("weights", save_name))
    test_acc, _, = evaluation(wandb.config, mmnet, test_loader, mode="test")
    print("Testing... episode:", epoch * wandb.config.num_train + i+1, "test acc:", test_acc, " max val acc:", max_acc, "max_test_acc:", max_test_acc)
    wandb.log({"test_acc":test_acc})

if __name__ == "__main__":
    train()
