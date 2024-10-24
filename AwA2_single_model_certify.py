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
from scipy.stats import norm

batch_size = 1
batch_size_certify = 1
num_certify = 100
num_test = 200
N_certify = 1000
noise_sd = 0.50
noise_sd_str = '0.50'
max_norm = 0.125

seed = 2023
alpha = 0.1

train_dataset = get_dataset('AWA', 'train')
test_dataset = get_dataset('AWA', 'test')

val_dataset, test_dataset,_ = random_split(test_dataset, [num_certify, num_test,len(test_dataset)-num_certify-num_test])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                          num_workers=8, pin_memory=False)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=8, pin_memory=False)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=8, pin_memory=False)

model = get_architecture('resnet50', 'AWA')
model_path = f'/data/common/AwA2/saved_models/noise_sd_{noise_sd_str}/main.pth.tar'
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
model.eval()

def get_avg_smoothed_score(Y, label):
    Y = Y.cpu()
    label = label.cpu()

    grey_box = ProbAccum(Y)
    rng = np.random.default_rng(seed)
    epsilon = rng.uniform(low=0.0, high=1.0, size=len(Y))
    alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
    # print(f'np.average(alpha_max): {np.average(alpha_max)}')
    return 1-np.average(alpha_max)

y_hat = []
labels = []
smoothed_score = torch.zeros((num_certify)).cuda()
with torch.no_grad():
    for i, (inputs, targets) in tqdm(enumerate(val_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        for k in range(N_certify // batch_size_certify):
            inputs = inputs.repeat((batch_size_certify, 1, 1, 1))
            targets = targets.repeat((batch_size_certify))
            noise = torch.randn_like(inputs).cuda() * noise_sd
            Y = model(inputs + noise).softmax(-1)
            smoothed_score[i] += get_avg_smoothed_score(Y, targets)
smoothed_score = smoothed_score.cpu().numpy()
smoothed_score = smoothed_score / (N_certify // batch_size_certify)

torch.save(smoothed_score, f'log/smoothed_score_single_model_AwA2')

smoothed_score = norm.ppf(smoothed_score)
level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(len(smoothed_score)))
score = mquantiles(smoothed_score, prob=level_adjusted)
score_worst_case = score - max_norm / noise_sd
score_worst_case = norm.cdf(score_worst_case)

ori = np.sort(smoothed_score)
total = len(ori)
covered = 0.
for x in ori:
    if x < score_worst_case:
        covered += 1
worst_case_coverage = covered / total
print(f'score_worst_case: {score_worst_case}')
print(f'worst case coverage: {worst_case_coverage}')


new_score = score + max_norm / noise_sd
new_score = norm.cdf(new_score)

# conformal prediction
alpha_calibrated = 1 - new_score
y_hat = []
labels = []

with torch.no_grad():
    for i, (inputs, targets) in tqdm(enumerate(test_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

        with torch.enable_grad():
            inputs = conformal_attack(inputs, targets, model, max_norm=max_norm)
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
