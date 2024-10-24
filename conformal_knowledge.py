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

def conformal_knowledge(args):
    print(f'Loading dataset')
    if args.dataset == 'GTSRB':
        data = DataMain(batch_size=args.batch_size)
        data.data_set_up(istrain=False)
        data.greeting()
        num_label, num_attribute, num_channel = 12, 13, 3

        mappings_label, mappings_attribute = get_knowledge(args.dataset)

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
            model.eval()
            model_list_label.append(model)
        for i in range(num_attribute):
            model = NEURAL(n_class=1, n_channel=num_channel)
            if os.path.exists("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, args.sigma)):
                model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, args.sigma)))
            else:
                model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f.pt" % (i, args.sigma)))
            model = model.cuda()
            model.eval()
            model_list_attribute.append(model)
    else:
        print(f'Dataset {args.dataset} is not implemented!')
        sys.exit(1)

    # Unioun bound
    args.alpha = args.alpha / (num_label + num_attribute)

    if args.method_conformal == 'split_conformal':
        if args.calibrate and not args.inference:
            alpha_calibrated_label = []
            alpha_calibrated_attribute = []

            # calibrate label sensors
            for i in tqdm(range(num_label)):
                model = model_list_label[i]
                label_mapping = mappings_label[i]

                cur = 0
                y_hat = torch.zeros((len(data.y_val), 2)).cuda()
                label = torch.zeros((len(data.y_val))).cuda()

                X, GT = data.sequential_val_batch()
                while X is not None:
                    X = X.cuda()  # to(device)
                    # X = X + torch.randn_like(X).cuda() * sigma
                    GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
                    Y = model(X)
                    this_batch_size = len(Y)
                    Y = Y.sigmoid()
                    for ii in range(Y.shape[0]):
                        sum_ = Y[ii].sum()
                        Y[ii] /= sum_
                    y_hat[cur:cur + this_batch_size, :] = Y
                    label[cur:cur + this_batch_size] = GT
                    cur = cur + this_batch_size
                    X, GT = data.sequential_val_batch()
                y_hat = y_hat.detach().cpu()
                label = label.cpu()

                n2 = y_hat.shape[0]
                grey_box = ProbAccum(y_hat)
                rng = np.random.default_rng(args.seed)
                epsilon = rng.uniform(low=0.0, high=1.0, size=n2)

                label = label.int()
                alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)

                scores = args.alpha - alpha_max
                level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
                alpha_correction = mquantiles(scores, prob=level_adjusted)
                alpha_calibrated_ = args.alpha - alpha_correction
                alpha_calibrated_label.append(alpha_calibrated_[0])

            # calibrate attribute sensors
            for i in tqdm(range(num_attribute)):
                model = model_list_attribute[i]
                label_mapping = mappings_attribute[i]

                cur = 0
                y_hat = torch.zeros((len(data.y_val), 2)).cuda()
                label = torch.zeros((len(data.y_val))).cuda()
                X, GT = data.sequential_val_batch()
                while X is not None:
                    X = X.cuda()  # to(device)
                    # X = X + torch.randn_like(X).cuda() * sigma
                    GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
                    Y = model(X)
                    this_batch_size = len(Y)
                    Y = Y.sigmoid()
                    for ii in range(Y.shape[0]):
                        sum_ = Y[ii].sum()
                        Y[ii] /= sum_
                    y_hat[cur:cur + this_batch_size, :] = Y
                    label[cur:cur + this_batch_size] = GT
                    cur = cur + this_batch_size
                    X, GT = data.sequential_val_batch()
                y_hat = y_hat.detach().cpu()
                label = label.cpu()

                n2 = y_hat.shape[0]
                grey_box = ProbAccum(y_hat)
                rng = np.random.default_rng(args.seed)
                epsilon = rng.uniform(low=0.0, high=1.0, size=n2)

                label = label.int()
                alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
                scores = args.alpha - alpha_max
                level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
                alpha_correction = mquantiles(scores, prob=level_adjusted)
                alpha_calibrated_ = args.alpha - alpha_correction
                alpha_calibrated_attribute.append(alpha_calibrated_[0])

            print(f'alpha_calibrated_label: {alpha_calibrated_label}')
            print(f'alpha_calibrated_attribute: {alpha_calibrated_attribute}')
            torch.save(alpha_calibrated_label, 'log/alpha_calibrated_label_knowledge')
            print(f'saving alpha_calibrated_label at log/alpha_calibrated_label_knowledge')
            torch.save(alpha_calibrated_attribute, 'log/alpha_calibrated_attribute_knowledge')
            print(f'saving alpha_calibrated_attribute at log/alpha_calibrated_attribute_knowledge')
        elif not args.calibrate and args.inference:

            if args.attack_type=='pgd':
                print('Performing pgd attack')
                X, GT = data.sequential_test_batch()
                cnt = 0
                while X is not None:
                    cnt += 1
                    print(f'pgd attack batch {cnt}')
                    X = X.cuda()
                    GT_attr = torch.from_numpy(mappings_attribute[:, GT.numpy()]).transpose(0, 1)
                    GT = GT.cuda()
                    GT_attr = GT_attr.cuda()
                    X = conformal_attack_knowledge(X, GT, model_list_label, GT_attr, model_list_attribute, max_norm=args.max_norm)
                    torch.save(X, f'log/adv_sample/adv_knowledge_{cnt}')
                    X, GT = data.sequential_test_batch()

            alpha_calibrated_label = torch.load('log/alpha_calibrated_label_knowledge')
            alpha_calibrated_attribute = torch.load('log/alpha_calibrated_attribute_knowledge')

            size_all = torch.zeros((len(data.y_test)))
            size_all = size_all + num_label
            label_test = data.y_test
            y_hat = {}
            for i in range(len(label_test)):
                y_hat[i] = list(range(0, num_label))

            # conformal inference on label sensors
            for i in tqdm(range(num_label)):
                model = model_list_label[i]
                label_mapping = mappings_label[i]

                cur = 0
                P_test = torch.zeros((len(data.y_test), 2)).cuda()
                X, GT = data.sequential_test_batch()
                cnt = 0
                while X is not None:
                    cnt += 1
                    X = X.cuda()  # to(device)
                    # X = X + torch.randn_like(X).cuda() * sigma
                    if args.attack_type=='pgd':
                        X = torch.load(f'log/adv_sample/adv_knowledge_{cnt}')
                    GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
                    Y = model(X)
                    this_batch_size = len(Y)
                    Y = Y.sigmoid()
                    for ii in range(Y.shape[0]):
                        sum_ = Y[ii].sum()
                        Y[ii] /= sum_
                    P_test[cur:cur + this_batch_size, :] = Y
                    cur = cur + this_batch_size
                    X, GT = data.sequential_test_batch()
                P_test = P_test.detach().cpu()

                rng = np.random.default_rng(args.seed)
                epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
                grey_box_test = ProbAccum(P_test)
                S_hat = grey_box_test.predict_sets(alpha_calibrated_label[i], epsilon=epsilon, allow_empty=False)

                for k, l in enumerate(S_hat):
                    # if size_all[k] == 1:
                    #     continue
                    if len(l) == 1 and l[0] == 0:
                        size_all[k] -= 1
                        y_hat[k].remove(i)
                    # elif len(l) == 1 and l[0] == 1:
                    #     size_all[k] = 1
                    #     y_hat[k] = [i]

            # conformal inference on attribute sensors
            for i in tqdm(range(num_attribute)):
                model = model_list_attribute[i]
                label_mapping = mappings_attribute[i]

                cur = 0
                P_test = torch.zeros((len(data.y_test), 2)).cuda()
                X, GT = data.sequential_test_batch()
                cnt = 0
                while X is not None:
                    cnt += 1
                    X = X.cuda()  # to(device)
                    # X = X + torch.randn_like(X).cuda() * sigma
                    if args.attack_type=='pgd':
                        X = torch.load(f'log/adv_sample/adv_knowledge_{cnt}')
                    GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
                    Y = model(X)
                    this_batch_size = len(Y)
                    Y = Y.sigmoid()
                    for ii in range(Y.shape[0]):
                        sum_ = Y[ii].sum()
                        Y[ii] /= sum_
                    P_test[cur:cur + this_batch_size, :] = Y
                    cur = cur + this_batch_size
                    X, GT = data.sequential_test_batch()
                P_test = P_test.detach().cpu()

                rng = np.random.default_rng(args.seed)
                epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
                grey_box_test = ProbAccum(P_test)
                S_hat = grey_box_test.predict_sets(alpha_calibrated_attribute[i], epsilon=epsilon, allow_empty=False)

                if args.knowledge_set_correction:
                    for k, l in enumerate(S_hat):
                        # if size_all[k] == 1:
                        #     continue
                        if len(l) == 1 and l[0] == 0:
                            for jj in range(len(mappings_attribute[i])):
                                if mappings_attribute[i][jj] == 1:
                                    if jj in y_hat[k]:
                                        size_all[k] -= 1
                                        y_hat[k].remove(jj)
                        elif len(l) == 1 and l[0] == 1:
                            for jj in range(len(mappings_attribute[i])):
                                if mappings_attribute[i][jj] == 0:
                                    if jj in y_hat[k]:
                                        size_all[k] -= 1
                                        y_hat[k].remove(jj)

            # evaluation
            avg_size = size_all.sum() / len(size_all)
            marginal_coverage = 0
            for i in range(len(label_test)):
                if label_test[i] in y_hat[i]:
                    marginal_coverage += 1
            marginal_coverage = marginal_coverage / len(label_test)
            print(f'marginal coverage under attack {args.attack_type}: {marginal_coverage}')
            print(f'avg_size under attack {args.attack_type}: {avg_size}')
        else:
            print(f'args.calibrate={args.calibrate}, args.inference={args.inference} are not satisfactory!')
            sys.exit(1)
    else:
        print(f'Conformal method {args.method_conformal} is not implemented!')
        sys.exit(1)