import torch
import scipy.stats
import numpy 
from dataloader.mini_imagenet import MiniImageNet
from dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm
import wandb

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def evaluation(config, model, data_loader, mode = "test", if_ssl = False):
    assert mode == "test" or mode == "val"
    num_total = 0
    num_way = 0
    num_query = 0
    num_shot = 0
    num_val = 0
    num_test = 0
    if if_ssl:
        num_way = config.ssl_num_way
        num_query = config.ssl_num_query
        num_shot = config.ssl_num_shot
        num_test = config.ssl_num_test
        num_val = config.ssl_num_val
    else:
        num_way = config.num_way
        num_query = config.num_query
        num_shot = config.num_shot
        num_test = config.num_test
        num_val = config.num_val
    if mode == "test":
        num_total = num_way * num_test * num_query

    if mode == "val":
        num_total = num_way * num_val * num_query

    ce = nn.CrossEntropyLoss().cuda()

    model.eval()

    acc_list = []

    label = torch.arange(num_way).repeat(num_query).cuda()
    save_name = str(config.model_type) + "_" + str(config.dataset) + "_" + str(config.proxy_type) + "_" + str(config.classifier) + "-" + str(config.optimizer) + "_" + str(config.temperature) + "_" + str(config.lam) + "*.txt"
    with torch.no_grad():
        total_rewards = 0
        total_loss = 0
        for batch, _, _ in tqdm(data_loader):
            data = batch.cuda()
            k = num_way * num_shot
            support, query = data[:k], data[k:]
            relation_score = 0
            relation_score, _, _, _ = model(support, query)
            loss = ce(-1 * relation_score, label)
            total_loss += loss.item()
            _, predict_label = torch.min(relation_score, 1)
            rewards = [1 if predict_label[j]==label[j] else 0 for j in range(label.shape[0])]
            total_rewards += numpy.sum(rewards)
            acc = numpy.sum(rewards) / 1.0 / num_way / num_query
            acc_list.append(acc)
            #if mode == "test":
            #    with open(save_name, "w") as f:
            #        for item in acc_list:
            #            f.write(str(item) + "\n")

        m, h = mean_confidence_interval(acc_list)
        print('Test mean accuracy of the model on the', mode, ' :{:.2f} %'.format(m * 100), "interval:", h * 100, "val ce loss:", total_loss / len(data_loader))
    return m, h
