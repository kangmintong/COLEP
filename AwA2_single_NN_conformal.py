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
from dataset.dataset import Cifar10
from model.model import NEURAL
from model.model_single import NEURAL_single
from tqdm import tqdm
from scipy.stats.mstats import mquantiles
import arc
from conformal_attack import conformal_attack
from model.resnet import resnet56
from RSCP import Score_Functions
from RSCP import utils
from dataset.datasets_AwA2 import get_dataset
from torch.utils.data import DataLoader
from model.architectures_AwA2 import get_architecture
import torch
from torch.nn import  Sigmoid, Softmax
from tqdm import tqdm
from torch.utils.data import random_split
from conformal_attack import conformal_attack

batch_size = 32
noise_sd = 0.25
max_norm = 0.25

seed = 2022
alpha = 0.1
calibration = False
inference = True

train_dataset = get_dataset('AWA', 'train')
test_dataset = get_dataset('AWA', 'test')

val_dataset, test_dataset = random_split(test_dataset, [6000, len(test_dataset)-6000])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                          num_workers=8, pin_memory=False)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=8, pin_memory=False)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=8, pin_memory=False)

model = get_architecture('resnet50', 'AWA')
model_path = f'/data/common/AwA2/saved_models/noise_sd_{noise_sd}/main.pth.tar'
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
model.eval()

if calibration:
    y_hat = []
    labels = []
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(val_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
            # compute output
            outputs = model(inputs).softmax(-1)
            y_hat.append(outputs)
            labels.append(targets)
    y_hat = torch.concat(y_hat, dim=0)
    labels = torch.concat(labels, dim=0)
    # print(y_hat[0])
    # print(y_hat[0].argmax(-1))
    # print(labels[0])
    y_hat = y_hat.detach().cpu()
    labels = labels.cpu()
    acc = torch.sum(y_hat.argmax(-1)==labels) / len(labels)
    print(f'Accuracy: {acc}')

    # conformal calibration
    n2 = y_hat.shape[0]
    grey_box = ProbAccum(y_hat)
    rng = np.random.default_rng(seed)
    epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
    alpha_max = grey_box.calibrate_scores(labels, epsilon=epsilon)
    scores = alpha - alpha_max
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
    alpha_correction = mquantiles(scores, prob=level_adjusted)
    alpha_calibrated = alpha - alpha_correction
    print(f'alpha_calibrated: {alpha_calibrated}')
    torch.save(alpha_calibrated, 'log/alpha_calibrated_model_single_awa')

if inference:
    # conformal prediction
    alpha_calibrated = torch.load('log/alpha_calibrated_model_single_awa')
    y_hat = []
    labels = []

    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            with torch.enable_grad():
                inputs = conformal_attack(inputs, targets, model, max_norm=max_norm, smoothadv=True)
            # compute output
            outputs = model(inputs).softmax(-1)
            y_hat.append(outputs)
            labels.append(targets)

    y_hat = torch.concat(y_hat, dim=0)
    labels = torch.concat(labels, dim=0)
    y_hat = y_hat.detach().cpu()
    labels = labels.cpu()

    rng = np.random.default_rng(seed)
    epsilon = rng.uniform(low=0.0, high=1.0, size=len(y_hat))
    grey_box_test = ProbAccum(y_hat)
    S_hat = grey_box_test.predict_sets(alpha_calibrated, epsilon=epsilon, allow_empty=False)

    # evaluation
    total_size = 0
    marginal_coverage = 0
    for i, l in enumerate(S_hat):
        total_size += len(l)
        if labels[i] in l:
            marginal_coverage += 1
    avg_size = total_size / len(S_hat)
    marginal_coverage = marginal_coverage / len(S_hat)

    print(f'marginal coverage: {marginal_coverage}')
    print(f'avg_size: {avg_size}')



