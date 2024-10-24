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

def conformal_baseline(args):
    print(f'Loading dataset')
    if args.dataset=='GTSRB':
        data = DataMain(batch_size=args.batch_size)
        data.data_set_up(istrain=False)
        data.greeting()
        num_label, num_channel = 12, 3

        print(f'Loading models')
        model = NEURAL_single(n_class=num_label, n_channel=num_channel)
        path = os.path.join(args.model_path, 'main/model_%.2f.pt' % (args.sigma))
        model.load_state_dict(torch.load(path))
        model = model.cuda()
        model.eval()
    elif args.dataset=='cifar10':
        data = Cifar10(batch_size=args.batch_size)
        data.data_set_up(istrain=False)
        data.greeting()
        num_label, num_channel = 10, 3

        print(f'Loading models')
        model = resnet56()
        path = f'pretrained_models/cifar10/model_single_cifar10_{args.sigma}'
        model= torch.load(path)
        model = model.cuda()
        model.n_class = num_label
        model.eval()
    else:
        print(f'Dataset {args.dataset} is not implemented!')
        sys.exit(1)


    if args.method_conformal == 'split_conformal':
        if args.calibrate and not args.inference:
            y_hat = torch.zeros((len(data.y_val), num_label)).cuda()
            label = data.y_val.cuda()
            X, GT = data.sequential_val_batch()
            cur = 0
            while X is not None:
                X = X.cuda()
                # X = X + torch.randn_like(X).cuda() * sigma
                Y = model(X)
                this_batch_size = len(Y)
                y_hat[cur:cur + this_batch_size, :] = Y.sigmoid()
                cur = cur + this_batch_size
                X, GT = data.sequential_val_batch()
            for i in range(y_hat.shape[0]):
                sum_ = y_hat[i].sum()
                y_hat[i] /= sum_
            y_hat = y_hat.detach().cpu()
            label = label.cpu()
            print(f'Accuracy: {(torch.argmax(y_hat,dim=1)==label).sum()/len(label)}')

            # conformal calibration
            n2 = y_hat.shape[0]
            if args.score_type=='aps':
                grey_box = ProbAccum(y_hat)
                rng = np.random.default_rng(args.seed)
                epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
                alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
                scores = args.alpha - alpha_max
                level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
                alpha_correction = mquantiles(scores, prob=level_adjusted)
                alpha_calibrated = args.alpha - alpha_correction
            elif args.score_type=='hps':
                scores = torch.zeros(n2)
                for i in range(n2):
                    scores[i] = 1 - y_hat[i][label[i]]
                # scores = args.alpha - scores
                level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
                alpha_correction = mquantiles(scores, prob=level_adjusted)
                alpha_calibrated = alpha_correction

            print(f'alpha_calibrated: {alpha_calibrated}')
            torch.save(alpha_calibrated, 'log/alpha_calibrated_model_single')
            print(f'saving alpha_calibrated at log/alpha_calibrated_model_single')

        elif not args.calibrate and args.inference:

            alpha_calibrated = torch.load('log/alpha_calibrated_model_single')

            P_test = torch.zeros((len(data.y_test), num_label)).cuda()
            label_test = data.y_test

            if args.attack_type=='physical_attack':
                data.X_test = np.load('./data/physical_attack/stop_sign_adv_X_test.npy')
                data.y_test = np.load('./data/physical_attack/stop_sign_label.npy')
                data.X_test = np.array([data.pre_process_image(data.X_test[i]) for i in range(len(data.X_test))],dtype=np.float32)
                data.y_test = np.array(data.y_test, dtype=np.long)
                data.X_test = torch.FloatTensor(data.X_test)
                data.X_test = data.X_test.permute(0, 3, 1, 2)
                data.y_test = torch.LongTensor(data.y_test)

            cur = 0
            X, GT = data.sequential_test_batch()
            while X is not None:
                X = X.cuda()
                # X = X + torch.randn_like(X).cuda() * sigma
                GT = GT.cuda()

                if args.attack_type=='pgd':
                    X = conformal_attack(X, GT, model, max_norm=args.max_norm)
                elif args.attack_type=='smoothadv':
                    X = conformal_attack(X, GT, model, max_norm=args.max_norm, smoothadv=True)
                Y = model(X)
                this_batch_size = len(Y)
                P_test[cur:cur + this_batch_size, :] = Y.sigmoid()
                cur = cur + this_batch_size
                X, GT = data.sequential_test_batch()

            for i in range(P_test.shape[0]):
                sum_ = P_test[i].sum()
                P_test[i] /= sum_
            P_test = P_test.detach().cpu()

            if args.score_type == 'aps':
                rng = np.random.default_rng(args.seed)
                epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
                grey_box_test = ProbAccum(P_test)
                S_hat = grey_box_test.predict_sets(alpha_calibrated, epsilon=epsilon, allow_empty=False)
            elif args.score_type == 'hps':
                S_hat = []
                n2 = len(P_test)
                for i in range(n2):
                    tmp = []
                    for j in range(num_label):
                        if 1 - P_test[i][j].item() <= alpha_calibrated:
                            tmp.append(j)
                    S_hat.append(tmp)

            # evaluation
            total_size = 0
            marginal_coverage = 0
            for i, l in enumerate(S_hat):
                total_size += len(l)
                if label_test[i] in l:
                    marginal_coverage += 1
            avg_size = total_size / len(S_hat)
            marginal_coverage = marginal_coverage / len(S_hat)

            print(f'marginal coverage with attack type {args.attack_type}: {marginal_coverage}')
            print(f'avg_size with attack type {args.attack_type}: {avg_size}')
        else:
            print(f'args.calibrate={args.calibrate}, args.inference={args.inference} are not satisfactory!')
            sys.exit(1)
    else:
        print(f'Conformal method {args.method_conformal} is not implemented!')
        sys.exit(1)