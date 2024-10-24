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

from conformal_baseline import conformal_baseline
from conformal_binary import conformal_binary
from conformal_knowledge import conformal_knowledge
from conformal_knowledge_pc import conformal_knowledge_pc
from conformal_knowledge_pc_certified_robustness import conformal_knowledge_pc_certified_robustness
from conformal_baseline_certified_robustness import conformal_baseline_certified_robustness
from conformal_knowledge_pc_ori_score import conformal_knowledge_pc_ori_score
from conformal_knowledge_pc_certified_robustness_ori_score import conformal_knowledge_pc_certified_robustness_ori_score
from conformal_knowledge_pc_worst_cov import conformal_knowledge_pc_worst_cov

def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, choices=['conformal_baseline','conformal_binary','conformal_knowledge','conformal_knowledge_pc','conformal_knowledge_pc_ori_score','conformal_knowledge_pc_worst_cov','train_PC'], help="type of method")
    parser.add_argument("-score_type", type=str, choices=['hps','aps'], help="type of scores", default='aps')
    parser.add_argument("-knowledge_set_correction", type=int, default=0)
    parser.add_argument("-pc_correction", type=int, default=1)
    parser.add_argument("-pc_weight", type=float, default=0.5)
    parser.add_argument("-knowledge_weights", type=float, default=1.0)
    parser.add_argument("-method_conformal", type=str, choices=['split_conformal'], default='split_conformal', help="type of conformal")
    parser.add_argument("-calibrate", action='store_true')
    parser.add_argument("-inference", action='store_true')
    parser.add_argument("-alpha", type=float, default=0.1)
    parser.add_argument("-load_exist", type=int, default=0)

    parser.add_argument("-dataset", type=str, default='GTSRB')
    parser.add_argument("-batch_size", type=int, default=400)
    parser.add_argument("-model_path", type=str, default='pretrained_models/')
    parser.add_argument("-sigma", type=float, default=0.12, help="smooth std of the pretrained models")

    parser.add_argument("-attack_type", type=str, choices=['none','pgd','certify','physical_attack','smoothadv'], help='attack type of conformal prediction')
    parser.add_argument("-max_norm", type=float, default=0.125, help='maximal l2 norm attack')
    parser.add_argument("-sigma_certify", type=float, default=0.5)
    parser.add_argument("-N_certify", type=int, default=100000)
    parser.add_argument("-certify_batchsize", type=int, default=10000)
    parser.add_argument("-alpha_certify", type=float, default=0.001)
    parser.add_argument("-skip_certify", type=int, default=10)
    parser.add_argument("-num_certify", type=int, default=1)


    parser.add_argument("-seed", type=int, default=2023)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = set_params()

    if args.method=='conformal_baseline':
        print(f'Using method {args.method}')
        if args.attack_type != 'certify':
            conformal_baseline(args)
        elif args.attack_type == 'certify':
            conformal_baseline_certified_robustness(args)
    elif args.method=='conformal_binary' and args.attack_type!='certify':
        print(f'Using method {args.method}')
        conformal_binary(args)
    elif args.method=='conformal_knowledge' and args.attack_type!='certify':
        print(f'Using method {args.method}')
        conformal_knowledge(args)
    elif args.method=='conformal_knowledge_pc':
        print(f'Using method {args.method}')
        if args.attack_type != 'certify':
            conformal_knowledge_pc(args)
        elif args.attack_type == 'certify':
            conformal_knowledge_pc_certified_robustness(args)
    elif args.method=='conformal_knowledge_pc_ori_score':
        print(f'Using method {args.method}')
        if args.attack_type != 'certify':
            conformal_knowledge_pc_ori_score(args)
        elif args.attack_type == 'certify':
            conformal_knowledge_pc_certified_robustness_ori_score(args)
    elif args.method=='conformal_knowledge_pc_worst_cov':
        print(f'Using method {args.method}')
        conformal_knowledge_pc_worst_cov(args)
    elif args.method=='train_PC':
        from train_PC import train_pc
        print(f'Using method {args.method}')
        train_pc(args)
    else:
        print(f'Method of {args.method} and attack type {args.attack_type} is not implemented!')
        sys.exit(1)