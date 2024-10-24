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
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
import time
from dataset.dataset import Cifar10
from model.resnet import resnet56

def get_avg_smoothed_score(Y, label, args):
    Y = Y.cpu()
    label = label.cpu()
    if args.score_type=='aps':
        # print(f'Y.shape:{Y.shape}')
        # print(f'label.shape:{label.shape}')
        grey_box = ProbAccum(Y)
        rng = np.random.default_rng(args.seed)
        epsilon = rng.uniform(low=0.0, high=1.0, size=len(Y))
        alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
        # print(f'np.average(alpha_max): {np.average(alpha_max)}')
        return 1-np.average(alpha_max)
    elif args.score_type=='hps':
        scores = torch.zeros(len(Y))
        for i in range(len(Y)):
            scores[i] = Y[i][label[i]]
        # scores = args.alpha - scores
        level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(len(Y)))
        alpha_correction = mquantiles(scores, prob=level_adjusted)
        alpha_calibrated = alpha_correction
        return np.average(alpha_calibrated)

def get_upper_bound_smoothed_score(smoothed_score, N_certify, delta):
    # TODO: should use the concentration bound more rigorously
    return smoothed_score + delta


def conformal_baseline_certified_robustness(args):
    print(f'Loading dataset')
    if args.dataset == 'GTSRB':
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
    elif args.dataset == 'cifar10':
        data = Cifar10(batch_size=args.batch_size)
        data.data_set_up(istrain=False)
        data.greeting()
        num_label, num_channel = 10, 3

        print(f'Loading models')
        model = resnet56()
        path = f'pretrained_models/cifar10/model_single_cifar10_{args.sigma}'
        model = torch.load(path)
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
            # while X is not None:
            #     X = X.cuda()
            #     # X = X + torch.randn_like(X).cuda() * sigma
            #     Y = model(X)
            #     this_batch_size = len(Y)
            #     y_hat[cur:cur + this_batch_size, :] = Y.sigmoid()
            #     cur = cur + this_batch_size
            #     X, GT = data.sequential_val_batch()
            # for i in range(y_hat.shape[0]):
            #     sum_ = y_hat[i].sum()
            #     y_hat[i] /= sum_
            # y_hat = y_hat.detach().cpu()
            # label = label.cpu()
            # print(f'Accuracy: {(torch.argmax(y_hat, dim=1) == label).sum() / len(label)}')

            # conformal calibration
            # n2 = y_hat.shape[0]
            # grey_box = ProbAccum(y_hat)
            # rng = np.random.default_rng(args.seed)
            # epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
            # alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
            #
            # scores = args.alpha - alpha_max
            # level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
            # alpha_correction = mquantiles(scores, prob=level_adjusted)
            # alpha_calibrated = args.alpha - alpha_correction


            # randomized smoothing for the conformity score of a single model
            smoothed_score = torch.zeros((args.num_certify)).cuda()
            delta = args.max_norm / args.sigma_certify

            X, GT = data.sequential_val_batch()
            cnt = 0
            num_certified = 0
            start_time = time.time()
            with torch.no_grad():
                while X is not None:
                    cnt += 1
                    if cnt % args.skip_certify and num_certified < args.num_certify:
                        X = X.cuda()
                        for k in range(args.N_certify // args.certify_batchsize):
                            # print(f'X.shape:{X.shape}')
                            # print(f'args.certify_batchsize:{args.certify_batchsize}')
                            X_ = X.repeat((args.certify_batchsize, 1, 1, 1))
                            GT_ = GT.repeat((args.certify_batchsize))
                            noise = torch.randn_like(X_).cuda() * args.sigma_certify
                            Y = model(X_ + noise).softmax(-1)
                            smoothed_score[num_certified] += get_avg_smoothed_score(Y, GT_, args)
                        num_certified += 1
                        cur_time = time.time()
                        print(f'certify {num_certified} samples, wall clock time {cur_time - start_time}')
                    X, GT = data.sequential_val_batch()

            smoothed_score = smoothed_score.cpu().numpy()
            smoothed_score = smoothed_score / (args.N_certify // args.certify_batchsize)
            print(f'smoothed_score_before: {torch.from_numpy(smoothed_score)}')
            smoothed_score = norm.ppf(smoothed_score)


            # scores = args.alpha - smoothed_score
            level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(len(smoothed_score)))
            alpha_calibrated = mquantiles(smoothed_score, prob=level_adjusted)
            # alpha_calibrated = args.alpha - alpha_correction

            print(f'alpha_calibrated_before: {alpha_calibrated}')
            alpha_calibrated = get_upper_bound_smoothed_score(alpha_calibrated, args.N_certify, delta)
            print(f'alpha_calibrated_after: {alpha_calibrated}')
            alpha_calibrated = norm.cdf(alpha_calibrated)
            # print(f'smoothed_score_after: {smoothed_score}')

            print(f'alpha_calibrated: {alpha_calibrated}')
            torch.save(alpha_calibrated, f'log/alpha_calibrated_model_single_certify_{args.sigma}')
            print(f'saving alpha_calibrated at log/alpha_calibrated_model_single_certify_{args.sigma}')

        elif not args.calibrate and args.inference:

            alpha_calibrated = torch.load(f'log/alpha_calibrated_model_single_certify_{args.sigma}')

            P_test = torch.zeros((len(data.y_test), num_label)).cuda()
            label_test = data.y_test

            cur = 0
            X, GT = data.sequential_test_batch()
            while X is not None:
                X = X.cuda()
                # X = X + torch.randn_like(X).cuda() * sigma
                GT = GT.cuda()

                # if args.attack_type == 'pgd':
                #     X = conformal_attack(X, GT, model, max_norm=args.max_norm)
                # elif args.attack_type == 'smoothadv':
                X = conformal_attack(X, GT, model, max_norm=args.max_norm, smoothadv=False)

                Y = model(X)
                this_batch_size = len(Y)
                P_test[cur:cur + this_batch_size, :] = Y.softmax(-1)
                cur = cur + this_batch_size
                X, GT = data.sequential_test_batch()

            # for i in range(P_test.shape[0]):
            #     sum_ = P_test[i].sum()
            #     P_test[i] /= sum_
            P_test = P_test.detach().cpu()

            # rng = np.random.default_rng(args.seed)
            # epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
            # grey_box_test = ProbAccum(P_test)
            # S_hat = grey_box_test.predict_sets(alpha_calibrated, epsilon=epsilon, allow_empty=False)

            if args.score_type == 'aps':
                rng = np.random.default_rng(args.seed)
                epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
                grey_box_test = ProbAccum(P_test)
                print(f'used conformity quantile: {alpha_calibrated}')
                S_hat = grey_box_test.predict_sets(1.-alpha_calibrated, epsilon=epsilon, allow_empty=False)
            elif args.score_type == 'hps':
                S_hat = []
                n2 = len(P_test)
                for i in range(n2):
                    tmp = []
                    for j in range(num_label):
                        if P_test[i][j].item() >= alpha_calibrated:
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

            print(f'marginal coverage of a single model with attack type {args.attack_type}: {marginal_coverage}')
            print(f'avg_size of a single model with attack type {args.attack_type}: {avg_size}')
        else:
            print(f'args.calibrate={args.calibrate}, args.inference={args.inference} are not satisfactory!')
            sys.exit(1)
    else:
        print(f'Conformal method {args.method_conformal} is not implemented!')
        sys.exit(1)