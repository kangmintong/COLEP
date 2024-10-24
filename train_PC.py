import argparse
from used_knowledge.used_knowledge import get_knowledge
import numpy as np
import pandas as pd
import os.path
from os import path
from dataset.dataset_conformal import GetDataset
import random
import torch
import sys
from arc.classification import ProbabilityAccumulator as ProbAccum
from dataset.dataset import DataMain
from model.model import NEURAL
from model.model_single import NEURAL_single
from tqdm import tqdm
from scipy.stats.mstats import mquantiles
import arc
from conformal_attack import conformal_attack
from conformal_attack import conformal_attack_knowledge
from knowledge_probabilistic_circuit.knwoledge_pc import knowledge_pc
from conformal_attack import conformal_attack_pc
from dataset.dataset import Cifar10
from model.resnet import resnet56
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from itertools import chain

def train_pc(args):
    print(f'Loading dataset')
    if args.dataset == 'GTSRB':
        data = DataMain(batch_size=args.batch_size)
        data.data_set_up(istrain=True)
        data.greeting()
        num_label, num_attribute, num_channel = 12, 13, 3

        mappings_label, mappings_attribute, rule_weight = get_knowledge(args.dataset, use_pc=True)

        print(f'Loading models')
        model_list_label = []
        model_list_attribute = []
        for i in range(num_label):
            model = NEURAL(n_class=1, n_channel=num_channel)
            if os.path.exists("pretrained_models/hier/model_%d_%.2f_5.pt" % (i, args.sigma)):
                model.load_state_dict(torch.load("pretrained_models/hier/model_%d_%.2f_5.pt" % (i, args.sigma)))
            else:
                model.load_state_dict(torch.load("pretrained_models/hier/model_%d_%.2f.pt" % (i, args.sigma)))
            model = model.cuda()
            model_list_label.append(model)
        for i in range(num_attribute):
            model = NEURAL(n_class=1, n_channel=num_channel)
            if os.path.exists("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, args.sigma)):
                model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, args.sigma)))
            else:
                model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f.pt" % (i, args.sigma)))
            model = model.cuda()
            model_list_attribute.append(model)

        knowledge = np.transpose(mappings_attribute)
        rule_weight = np.transpose(rule_weight)
        knowledge = torch.from_numpy(knowledge).cuda()
        knowledge_weight = torch.zeros_like(knowledge)
        for i in range(knowledge.shape[0]):
            for j in range(knowledge.shape[1]):
                if knowledge[i][j] == 1:
                    knowledge_weight[i][j] = rule_weight[i][j] * args.knowledge_weights
        model_pc_reasoning = knowledge_pc(model_list_label, model_list_attribute, knowledge, knowledge_weight, args, train=True)
    params = None
    for i in range(num_label):
        if params == None:
            params = list(model_pc_reasoning.label_sensor_list[i].parameters())
        else:
            params = params + list(model_pc_reasoning.label_sensor_list[i].parameters())
    for i in range(num_attribute):
        params = params + list(model_pc_reasoning.attr_sensor_list[i].parameters())
    optimizer = optim.SGD(params, lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=False)

    loss_f = F.cross_entropy

    epochs = 30
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs/3), int(epochs/3*2)], gamma=0.1)

    model_pc_reasoning.args.pc_correction=0
    ori_weight = model_pc_reasoning.knowledge_weight
    for i in tqdm(range(epochs)):
        X, GT = data.sequential_val_batch()
        cnt = 0
        loss_avg = 0.
        while X is not None:
            cnt += 1
            X = X.cuda()
            GT_attr = torch.from_numpy(mappings_attribute[:, GT.numpy()]).transpose(0, 1)
            GT = GT.cuda()
            GT_attr = GT_attr.cuda()

            X = conformal_attack_pc(X, model_pc_reasoning, GT, GT_attr, max_norm=args.max_norm, num_class=num_label)

            Y = model_pc_reasoning(X)
            # Y = Y[:,:,1]
            # loss = loss_f(Y, GT)

            ce_loss = F.cross_entropy(Y[:, :num_label, 1], GT, reduction='mean')
            loss = ce_loss

            bce_loss = F.binary_cross_entropy_with_logits(Y[:, num_label:, 1], GT_attr.float(), reduction='mean')
            loss += bce_loss
            # print(torch.autograd.grad(loss, [model_pc_reasoning.knowledge_weight],allow_unused=True)[0].detach())
            loss_avg += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(model_pc_reasoning.knowledge_weight.grad)
            X, GT = data.sequential_val_batch()
        print(f'loss_avg at iteration {i}: {loss_avg/cnt}')
        lr_scheduler.step()
    model_pc_reasoning.args.pc_correction = 1
    # print(model_pc_reasoning.knowledge_weight - ori_weight)
    torch.save(model_pc_reasoning,f'log/model_pc_reasonsing_{args.dataset}_{args.sigma}')

