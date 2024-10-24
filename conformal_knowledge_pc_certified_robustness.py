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


def conformal_knowledge_pc_certified_robustness(args):
    print(f'Loading dataset')
    if args.dataset == 'GTSRB':
        data = DataMain(batch_size=args.batch_size)
        data.data_set_up(istrain=False)
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

        knowledge = np.transpose(mappings_attribute)
        rule_weight = np.transpose(rule_weight)
        knowledge = torch.from_numpy(knowledge).cuda()
        knowledge_weight = torch.zeros_like(knowledge)
        for i in range(knowledge.shape[0]):
            for j in range(knowledge.shape[1]):
                if knowledge[i][j] == 1:
                    knowledge_weight[i][j] = rule_weight[i][j] * args.knowledge_weights
        model_pc_reasoning = knowledge_pc(model_list_label, model_list_attribute, knowledge, knowledge_weight, args)
    elif args.dataset == 'cifar10':
        data = Cifar10(batch_size=args.batch_size)
        data.data_set_up(istrain=False)
        data.greeting()
        num_label, num_attribute, num_channel = 10, 9, 3

        mappings_label, mappings_attribute, rule_weight = get_knowledge(args.dataset, use_pc=True)

        print(f'Loading models')
        model_list_label = []
        model_list_attribute = []
        for i in range(num_label):
            model = resnet56(n_class=1)
            model = torch.load(f'pretrained_models/cifar10/model_sensor_{i}_cifar10_{args.sigma}')
            model = model.cuda()
            model.eval()
            model_list_label.append(model)
        for i in range(num_attribute):
            model = resnet56(n_class=1)
            model = torch.load(f'pretrained_models/cifar10/model_sensor_{i+10}_cifar10_{args.sigma}')
            model = model.cuda()
            model.eval()
            model_list_attribute.append(model)

        knowledge = np.transpose(mappings_attribute)
        rule_weight = np.transpose(rule_weight)
        knowledge = torch.from_numpy(knowledge).cuda()
        knowledge_weight = torch.zeros_like(knowledge)
        for i in range(knowledge.shape[0]):
            for j in range(knowledge.shape[1]):
                if knowledge[i][j] == 1:
                    knowledge_weight[i][j] = rule_weight[i][j] * args.knowledge_weights
        model_pc_reasoning = knowledge_pc(model_list_label, model_list_attribute, knowledge, knowledge_weight, args)
    else:
        print(f'Dataset {args.dataset} is not implemented!')
        sys.exit(1)

    # Unioun bound
    args.alpha = (args.alpha -args.alpha_certify*(num_label+num_attribute)) / (num_label + num_attribute)

    if args.method_conformal == 'split_conformal':
        if args.calibrate and not args.inference:
            alpha_calibrated_label = []
            alpha_calibrated_attribute = []

            y_hat_all = torch.zeros((len(data.y_val), num_label + num_attribute, 2)).cuda()
            X, GT = data.sequential_val_batch()
            cur = 0
            with torch.no_grad():
                while X is not None:
                    X = X.cuda()
                    Y = model_pc_reasoning(X)
                    this_batch_size = len(Y)
                    y_hat_all[cur:cur + this_batch_size, :, :] = Y
                    cur = cur + this_batch_size
                    X, GT = data.sequential_val_batch()

            print(
                f'Accuracy: {(torch.argmax(y_hat_all[:, :num_label, 1].detach().cpu(), dim=1) == data.y_val).sum() / len(data.y_val)}')

            # calibrate label sensors
            for i in tqdm(range(num_label)):
                label_mapping = mappings_label[i]
                cur = 0
                y_hat = y_hat_all[:, i]
                label = torch.zeros((len(data.y_val))).cuda()
                X, GT = data.sequential_val_batch()
                while X is not None:
                    X = X.cuda()
                    GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
                    this_batch_size = len(X)
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
                label_mapping = mappings_attribute[i]
                cur = 0
                y_hat = y_hat_all[:, i + num_label]
                label = torch.zeros((len(data.y_val))).cuda()

                X, GT = data.sequential_val_batch()
                while X is not None:
                    X = X.cuda()  # to(device)
                    GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
                    this_batch_size = len(X)
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
            torch.save(alpha_calibrated_label, f'log/alpha_calibrated_label_knowledge_pc_{args.pc_correction}_{args.sigma}_{args.attack_type}_{args.pc_weight}_{args.knowledge_weights}')
            print(f'alpha_calibrated_attribute: {alpha_calibrated_attribute}')
            torch.save(alpha_calibrated_attribute, f'log/alpha_calibrated_attribute_knowledge_pc_{args.pc_correction}_{args.sigma}_{args.attack_type}_{args.pc_weight}_{args.knowledge_weights}')
            print(f'saving alpha_calibrated_label at log/alpha_calibrated_label_knowledge_pc_{args.pc_correction}_{args.sigma}_{args.attack_type}_{args.pc_weight}_{args.knowledge_weights}')
            print(f'saving alpha_calibrated_attribute at log/alpha_calibrated_attribute_knowledge_pc_{args.pc_correction}_{args.sigma}_{args.attack_type}_{args.pc_weight}_{args.knowledge_weights}')
        elif not args.calibrate and args.inference:
            alpha_calibrated_label = torch.load(f'log/alpha_calibrated_label_knowledge_pc_{args.pc_correction}_{args.sigma}_{args.attack_type}_{args.pc_weight}_{args.knowledge_weights}')
            alpha_calibrated_attribute = torch.load(f'log/alpha_calibrated_attribute_knowledge_pc_{args.pc_correction}_{args.sigma}_{args.attack_type}_{args.pc_weight}_{args.knowledge_weights}')

            # randomized smoothing for each sensor
            delta = args.max_norm / args.sigma_certify
            label_test = []
            start_time = time.time()
            with torch.no_grad():
                    X, GT = data.sequential_test_batch()
                    cnt = 0
                    num_certified = 0
                    pAs = np.zeros((args.num_certify, num_label + num_attribute, 2))
                    sensor_upbnd = np.zeros((args.num_certify, num_label + num_attribute))
                    sensor_lowbnd = np.zeros((args.num_certify, num_label + num_attribute))
                    while X is not None:
                        cnt += 1
                        if cnt % args.skip_certify and num_certified < args.num_certify:
                            X = X.cuda()
                            label_test.append(GT)
                            for i in range(num_label+num_attribute):
                                if i<num_label:
                                    model = model_list_label[i]
                                else:
                                    model = model_list_attribute[i-num_label]
                                counts = model_pc_reasoning.smoothed_forward(model, X, args.sigma_certify, N=args.N_certify, pc_correction=args.pc_correction)
                                pos = counts.argmax(0)
                                pa = proportion_confint(counts[pos].item(), args.N_certify, alpha=2 * args.alpha_certify, method="beta")
                                if pos==1:
                                    sensor_upbnd[num_certified,i] = norm.cdf(norm.ppf(pa[1]) + delta)
                                    sensor_lowbnd[num_certified,i] = norm.cdf(norm.ppf(pa[0]) - delta)
                                else:
                                    sensor_upbnd[num_certified,i] = norm.cdf(norm.ppf(1-pa[0]) + delta)
                                    sensor_lowbnd[num_certified,i] = norm.cdf(norm.ppf(1-pa[1]) - delta)
                            num_certified += 1
                            cur_time = time.time()
                            print(f'certify {num_certified} samples, wall clock time {cur_time-start_time}')
                        X, GT = data.sequential_test_batch()
                    sensor_upbnd[sensor_upbnd>1.0] = 1.0
                    sensor_lowbnd[sensor_lowbnd<0.0] = 0.0

            # print('sensor_upbnd-sensor_lowbnd>=1e-20:')
            # print((sensor_upbnd-sensor_lowbnd>=1e-20))

            # certification for the PC part
            start_time_certify = time.time()
            sensor_lowbnd_pc_correction, sensor_upbnd_pc_correction = model_pc_reasoning.certify_PC(sensor_lowbnd[:,:num_label], sensor_upbnd[:,:num_label], sensor_lowbnd[:,num_label:], sensor_upbnd[:,num_label:], pc_correction=args.pc_correction)
            end_time_certify = time.time()
            print(f'certification time of {len(sensor_lowbnd_pc_correction)} samples: {end_time_certify-start_time_certify}')

            # print('(sensor_upbnd_pc_correction-sensor_lowbnd_pc_correction)>=-1e-20')
            # print((sensor_upbnd_pc_correction-sensor_lowbnd_pc_correction)>=-1e-20)


            # construct certified robust prediction set
            certified_prediction_set = {}
            for i in range(len(sensor_lowbnd_pc_correction)):
                certified_prediction_set[i] = list(range(0, num_label))
            y_hat_all = np.zeros((num_certified, num_label + num_attribute, 2))
            for i in range(num_certified):
                for j in range(num_label+num_attribute):
                    if sensor_lowbnd_pc_correction[i,j] > 0.5:
                        y_hat_all[i,j,1] = sensor_lowbnd_pc_correction[i,j]
                        y_hat_all[i,j,0] = 1-sensor_lowbnd_pc_correction[i,j]
                    elif sensor_upbnd_pc_correction[i,j] < 0.5:
                        y_hat_all[i, j, 1] = sensor_upbnd_pc_correction[i, j]
                        y_hat_all[i, j, 0] = 1 - sensor_upbnd_pc_correction[i, j]
                    else:
                        y_hat_all[i, j, 0] = y_hat_all[i, j, 1] = 0.5

            for i in range(num_label):
                P_test = y_hat_all[:, i, :]
                rng = np.random.default_rng(args.seed)
                epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
                grey_box_test = ProbAccum(P_test)
                S_hat = grey_box_test.predict_sets(alpha_calibrated_label[i], epsilon=epsilon, allow_empty=False)

                for k, l in enumerate(S_hat):
                    if len(l) == 1 and l[0] == 0:
                        certified_prediction_set[k].remove(i)

            # print(f'certified_prediction_set before knowledge_set_correction: {certified_prediction_set}')

            if args.knowledge_set_correction:
                for i in range(num_attribute):
                    P_test = y_hat_all[:, i + num_label, :]
                    rng = np.random.default_rng(args.seed)
                    epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
                    grey_box_test = ProbAccum(P_test)
                    S_hat = grey_box_test.predict_sets(alpha_calibrated_attribute[i], epsilon=epsilon, allow_empty=False)
                    for k, l in enumerate(S_hat):
                        # if size_all[k] == 1:
                        #     continue
                        if len(l) == 1 and l[0] == 0:
                            for jj in range(len(mappings_attribute[i])):
                                if mappings_attribute[i][jj] == 1:
                                    if jj in certified_prediction_set[k]:
                                        certified_prediction_set[k].remove(jj)
                        elif len(l) == 1 and l[0] == 1:
                            for jj in range(len(mappings_attribute[i])):
                                if mappings_attribute[i][jj] == 0:
                                    if jj in certified_prediction_set[k]:
                                        certified_prediction_set[k].remove(jj)

                # for i in range(num_certified):
                #     for j in range(num_attribute):
                #         if sensor_upbnd_pc_correction[i,j+num_label]<0.5:
                #             for jj in range(len(mappings_attribute[j])):
                #                 if mappings_attribute[j][jj] == 1:
                #                     if jj in certified_prediction_set[i]:
                #                         certified_prediction_set[i].remove(jj)
                #         elif sensor_lowbnd_pc_correction[i,j+num_label]>0.5:
                #             for jj in range(len(mappings_attribute[j])):
                #                 if mappings_attribute[j][jj] == 0:
                #                     if jj in certified_prediction_set[i]:
                #                         certified_prediction_set[i].remove(jj)



            # evaluation
            size_all = 0
            for i in range(len(sensor_lowbnd_pc_correction)):
                size_all += len(certified_prediction_set[i])
            avg_size = 1.0 * size_all / len(sensor_lowbnd_pc_correction)
            marginal_coverage = 0
            for i in range(len(label_test)):
                if label_test[i] in certified_prediction_set[i]:
                    marginal_coverage += 1
            marginal_coverage = marginal_coverage / len(label_test)
            torch.save(certified_prediction_set, f'log/certified_set/certified_set_{args.sigma}_{args.sigma_certify}')
            print(f'marginal coverage under attack {args.attack_type} with sigma {args.sigma} with sigma_certify {args.sigma_certify} with pc_correction {args.pc_correction}: {marginal_coverage}')
            print(f'avg_size under attack {args.attack_type} with sigma {args.sigma} with sigma_certify {args.sigma_certify} with pc_correction {args.pc_correction}: {avg_size}')
            # print(f'Certified_prediction_set')
            # print(certified_prediction_set)
            # print(f'Label_test')
            # print(label_test)
        else:
            print(f'args.calibrate={args.calibrate}, args.inference={args.inference} are not satisfactory!')
            sys.exit(1)
    else:
        print(f'Conformal method {args.method_conformal} is not implemented!')
        sys.exit(1)