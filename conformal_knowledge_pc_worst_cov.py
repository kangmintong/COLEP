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
import time
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
import math

def conformal_knowledge_pc_worst_cov(args):
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
    # args.alpha = args.alpha / num_label

    if args.method_conformal == 'split_conformal':
        # randomized smoothing for each sensor
        delta = args.max_norm / args.sigma_certify
        label_test = []
        start_time = time.time()
        with torch.no_grad():
            X, GT = data.sequential_val_batch()
            cnt = 0
            num_certified = 0
            pAs = np.zeros((args.num_certify, num_label + num_attribute, 2))
            sensor_upbnd = np.zeros((args.num_certify, num_label + num_attribute))
            sensor_lowbnd = np.zeros((args.num_certify, num_label + num_attribute))
            pred_clean = np.zeros((args.num_certify, num_label + num_attribute, 2))
            while X is not None:
                cnt += 1
                if cnt % args.skip_certify and num_certified < args.num_certify:
                    X = X.cuda()
                    label_test.append(GT)
                    pred_clean[num_certified:num_certified+1,:,:] = model_pc_reasoning(X).detach().cpu().numpy()


                    probs = model_pc_reasoning.smoothed_forward_2(X, args.sigma_certify, N=args.N_certify,
                                                                 pc_correction=args.pc_correction).detach().cpu().numpy()

                    sensor_upbnd[num_certified, :] = norm.cdf(norm.ppf(probs[:,1]) + delta)
                    sensor_lowbnd[num_certified, :] = norm.cdf(norm.ppf(probs[:,1]) - delta)

                    # print(f'probs[:,1]: {probs[:,1]}')
                    # print(f'sensor_lowbnd[num_certified, :]: {sensor_lowbnd[num_certified, :]}')
                    # print(f'sensor_upbnd[num_certified, :]: {sensor_upbnd[num_certified, :]}')

                    num_certified += 1
                    cur_time = time.time()
                    print(f'certify {num_certified} samples, wall clock time {cur_time - start_time}')
                X, GT = data.sequential_val_batch()
            sensor_upbnd[sensor_upbnd > 1.0] = 1.0
            sensor_lowbnd[sensor_lowbnd < 0.0] = 0.0


        print(f'sensor_lowbnd[0, :num_label]: {sensor_lowbnd[0, :num_label]}')
        print(f'sensor_upbnd[0, :num_label]: {sensor_upbnd[0, :num_label]}')
        # certification for the PC part
        start_time_certify = time.time()
        sensor_lowbnd_pc_correction, sensor_upbnd_pc_correction = model_pc_reasoning.certify_PC(
            sensor_lowbnd[:, :num_label], sensor_upbnd[:, :num_label], sensor_lowbnd[:, num_label:],
            sensor_upbnd[:, num_label:], pc_correction=args.pc_correction)
        end_time_certify = time.time()
        print(f'certification time of {len(sensor_lowbnd_pc_correction)} samples: {end_time_certify - start_time_certify}')

        print(f'sensor_lowbnd_pc_correction[0, :num_label]: {sensor_lowbnd_pc_correction[0, :num_label]}')
        print(f'sensor_upbnd_pc_correction[0, :num_label]: {sensor_upbnd_pc_correction[0, :num_label]}')

        # print('(sensor_upbnd_pc_correction-sensor_lowbnd_pc_correction)>=-1e-20')
        # print((sensor_upbnd_pc_correction-sensor_lowbnd_pc_correction)>=-1e-20)


        certified_prediction_set = {}
        for i in range(len(sensor_lowbnd_pc_correction)):
            certified_prediction_set[i] = list(range(0, num_label))
        y_hat_all = np.zeros((num_certified, num_label, 2))
        for i in range(num_certified):
            for j in range(num_label):
                label = torch.from_numpy(mappings_label[j][np.array([label_test[i]])])
                if label==1:
                    y_hat_all[i, j, 1] = sensor_lowbnd_pc_correction[i, j]
                    y_hat_all[i, j, 0] = 1 - sensor_lowbnd_pc_correction[i, j]
                elif label==0:
                    y_hat_all[i, j, 1] = sensor_upbnd_pc_correction[i, j]
                    y_hat_all[i, j, 0] = 1 - sensor_upbnd_pc_correction[i, j]
                # else:
                #     y_hat_all[i, j, 0] = y_hat_all[i, j, 1] = 0.5

        score_worst_case = np.zeros((args.num_certify,num_label))
        score_clean = np.zeros((args.num_certify,num_label))
        miscov = 0.
        for i in range(num_label):
            y_hat = y_hat_all[:,i]
            y_hat_clean = pred_clean[:,i]
            label = torch.from_numpy(mappings_label[i][label_test])

            n2 = y_hat.shape[0]
            grey_box = ProbAccum(y_hat)
            rng = np.random.default_rng(args.seed)
            epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
            label = label.int()
            alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)

            grey_box = ProbAccum(y_hat_clean)
            alpha_max_clean = grey_box.calibrate_scores(label, epsilon=epsilon)

            alpha_max = np.array(alpha_max)
            alpha_max_clean = np.array(alpha_max_clean)

            n2 = len(alpha_max)
            level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
            alpha_ = mquantiles(-alpha_max, prob=level_adjusted)
            alpha_ = -alpha_[0]
            covered = 0
            for l in alpha_max_clean:
                if l > alpha_:
                    covered += 1
            covered = 1.0 * covered / len(alpha_max)
            miscov += 1 - covered

            print(f'miscoverage of label {i}: {1-covered}')

        print(f'certified miscoverage: {miscov}')



        #     scores = args.alpha - alpha_max
        #     level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
        #     alpha_correction = mquantiles(scores, prob=level_adjusted)
        #     alpha_calibrated_ = args.alpha - alpha_correction
        #
        #     alpha_calibrated_worst_case.append(alpha_calibrated_[0])
        #
        # print(alpha_calibrated_worst_case)



        # alpha_calibrated_label = []
        # alpha_calibrated_attribute = []
        #
        # data = DataMain(batch_size=200)
        # data.data_set_up(istrain=False)
        # y_hat_all = torch.zeros((len(data.y_val), num_label+num_attribute, 2)).cuda()
        # label = data.y_val
        # X, GT = data.sequential_val_batch()
        # cur = 0
        # with torch.no_grad():
        #     while X is not None:
        #         X = X.cuda()
        #         Y = model_pc_reasoning(X)
        #         this_batch_size = len(Y)
        #         y_hat_all[cur:cur + this_batch_size, :, :] = Y
        #         cur = cur + this_batch_size
        #         X, GT = data.sequential_val_batch()
        #
        # print(f'Accuracy: {(torch.argmax(y_hat_all[:,:num_label,1].detach().cpu(), dim=1) == data.y_val).sum() / len(data.y_val)}')
        # y_hat_all = y_hat_all[:,:num_label,1].softmax(-1).detach().cpu()
        #
        # data = DataMain(batch_size=args.batch_size)
        # data.data_set_up(istrain=False)
        #
        # # randomized smoothing for each sensor
        # delta = args.max_norm / args.sigma_certify
        #
        #
        #
        # label_test = []
        # start_time = time.time()
        #
        # with torch.no_grad():
        #     X, GT = data.sequential_val_batch()
        #     cnt = 0
        #     num_certified = 0
        #     # pAs = np.zeros((args.num_certify, num_label + num_attribute, 2))
        #     sensor_upbnd = np.zeros((args.num_certify, num_label + num_attribute))
        #     sensor_lowbnd = np.zeros((args.num_certify, num_label + num_attribute))
        #     pred = torch.zeros((num_certified,num_label + num_attribute)).cuda()
        #     while X is not None:
        #         cnt += 1
        #         if cnt % args.skip_certify and num_certified < args.num_certify:
        #             X = X.cuda()
        #             label_test.append(GT)
        #             N = args.N_certify
        #             batch_size = 5000
        #             with torch.no_grad():
        #                 for _ in range(math.ceil(args.N_certify / batch_size)):
        #                     this_batch_size = min(batch_size, N)
        #                     N -= this_batch_size
        #                     batch = X.repeat((this_batch_size, 1, 1, 1))
        #                     noise = torch.randn_like(batch, device='cuda') * args.sigma_certify
        #                     predictions = model_pc_reasoning(batch + noise)[:,:,1]
        #                     predictions = predictions.sigmoid()
        #                     pred[cnt,:] = pred[cnt,:] + predictions.mean(0) / math.ceil(args.N_certify / batch_size)
        #             num_certified += 1
        #             cur_time = time.time()
        #             print(f'certify {num_certified} samples, wall clock time {cur_time-start_time}, bound interval avg: {(sensor_upbnd[num_certified-1,:]-sensor_lowbnd[num_certified-1,:]).mean()}')
        #         X, GT = data.sequential_val_batch()
        #     sensor_upbnd[sensor_upbnd>1.0] = 1.0
        #     sensor_lowbnd[sensor_lowbnd<0.0] = 0.0
        #
        # print(f'mean bound interval before PC correction: {(sensor_upbnd - sensor_lowbnd).mean()}')
        # # certification for the PC part
        # start_time_certify = time.time()
        # sensor_lowbnd_pc_correction, sensor_upbnd_pc_correction = model_pc_reasoning.certify_PC(sensor_lowbnd[:,:num_label], sensor_upbnd[:,:num_label], sensor_lowbnd[:,num_label:], sensor_upbnd[:,num_label:], pc_correction=args.pc_correction)
        # end_time_certify = time.time()
        # print(f'certification time of {len(sensor_lowbnd_pc_correction)} samples: {end_time_certify-start_time_certify}')
        #
        # print(f'mean bound interval after PC correction: {(sensor_upbnd_pc_correction-sensor_lowbnd_pc_correction).mean()}')
        #
        # # construct certified robust prediction set
        # certified_prediction_set = {}
        # for i in range(len(sensor_lowbnd_pc_correction)):
        #     certified_prediction_set[i] = list(range(0, num_label))
        # y_hat_all = np.zeros((num_certified, num_label + num_attribute, 2))
        #
        # for i in range(num_certified):
        #     label_mapping = mappings_label[i]
        #     cur_label = torch.from_numpy(label_mapping[np.array([label_test[i]])])
        #     print(f'{sensor_lowbnd_pc_correction[i, 0]} -- {sensor_upbnd_pc_correction[i, 0]}')
        #     for j in range(num_label + num_attribute):
        #         if cur_label==1:
        #             y_hat_all[i, j, 1] = sensor_lowbnd_pc_correction[i, j]
        #             y_hat_all[i, j, 0] = 1 - sensor_lowbnd_pc_correction[i, j]
        #         elif cur_label==0:
        #             y_hat_all[i, j, 1] = sensor_upbnd_pc_correction[i, j]
        #             y_hat_all[i, j, 0] = 1 - sensor_upbnd_pc_correction[i, j]
        #         else:
        #             y_hat_all[i, j, 0] = y_hat_all[i, j, 1] = 0.5
        #
        # # conformal calibration
        # y_hat_all = y_hat_all[:,:,1]
        # for i in range(y_hat_all.shape[0]):
        #     y_hat_all[i] = np.exp(y_hat_all[i])
        #     sum_ = y_hat_all[i].sum()
        #     y_hat_all[i] /= sum_
        # label = np.array(label_test)
        #
        #
        # y_hat = y_hat_all
        # n2 = y_hat.shape[0]
        # if args.score_type == 'aps':
        #     grey_box = ProbAccum(y_hat)
        #     rng = np.random.default_rng(args.seed)
        #     epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        #     alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
        #     scores = args.alpha - alpha_max
        #     level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
        #     alpha_correction = mquantiles(scores, prob=level_adjusted)
        #     alpha_calibrated_label = args.alpha - alpha_correction
        # elif args.score_type == 'hps':
        #     scores = torch.zeros(n2)
        #     for i in range(n2):
        #         scores[i] = 1 - y_hat[i][label[i]]
        #     # scores = args.alpha - scores
        #     level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
        #     alpha_correction = mquantiles(scores, prob=level_adjusted)
        #     alpha_calibrated_label = alpha_correction
        #
        # # calibrate label sensors
        # # for i in tqdm(range(num_label)):
        # #     label_mapping = mappings_label[i]
        # #     cur = 0
        # #     y_hat = y_hat_all[:, i]
        # #     label = torch.from_numpy(label_mapping[label_test])
        # #     # X, GT = data.sequential_val_batch()
        # #     # while X is not None:
        # #     #     X = X.cuda()
        # #     #     GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
        # #     #     this_batch_size = len(X)
        # #     #     label[cur:cur + this_batch_size] = GT
        # #     #     cur = cur + this_batch_size
        # #     #     X, GT = data.sequential_val_batch()
        # #     # y_hat = y_hat
        # #     # label = label.cpu()
        # #
        # #     n2 = y_hat.shape[0]
        # #     grey_box = ProbAccum(y_hat)
        # #     rng = np.random.default_rng(args.seed)
        # #     epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        # #     label = label.int()
        # #
        # #     # print(f'n2: {}')
        # #     alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
        # #
        # #     scores = args.alpha - alpha_max
        # #     level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
        # #     alpha_correction = mquantiles(scores, prob=level_adjusted)
        # #     alpha_calibrated_ = args.alpha - alpha_correction
        # #     alpha_calibrated_label.append(alpha_calibrated_[0])
        #
        # # calibrate attribute sensors
        # # for i in tqdm(range(num_attribute)):
        # #     label_mapping = mappings_attribute[i]
        # #     cur = 0
        # #     y_hat = y_hat_all[:, i + num_label]
        # #     label = torch.zeros((len(data.y_val))).cuda()
        # #
        # #     X, GT = data.sequential_val_batch()
        # #     while X is not None:
        # #         X = X.cuda()  # to(device)
        # #         GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
        # #         this_batch_size = len(X)
        # #         label[cur:cur + this_batch_size] = GT
        # #         cur = cur + this_batch_size
        # #         X, GT = data.sequential_val_batch()
        # #     y_hat = y_hat.detach().cpu()
        # #     label = label.cpu()
        # #
        # #     n2 = y_hat.shape[0]
        # #     grey_box = ProbAccum(y_hat)
        # #     rng = np.random.default_rng(args.seed)
        # #     epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        # #     label = label.int()
        # #     alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
        # #
        # #     scores = args.alpha - alpha_max
        # #     level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
        # #     alpha_correction = mquantiles(scores, prob=level_adjusted)
        # #     alpha_calibrated_ = args.alpha - alpha_correction
        # #     alpha_calibrated_attribute.append(alpha_calibrated_[0])
        #
        # alpha_calibrated_label = alpha_calibrated_label
        # print(f'alpha_calibrated_label: {alpha_calibrated_label}')
        # torch.save(alpha_calibrated_label, f'log/alpha_calibrated_label_knowledge_pc_{args.dataset}')
        # print(f'alpha_calibrated_attribute: {alpha_calibrated_attribute}')
        # torch.save(alpha_calibrated_attribute, f'log/alpha_calibrated_attribute_knowledge_pc_{args.dataset}')
        # print(f'saving alpha_calibrated_label at log/alpha_calibrated_label_knowledge_pc_{args.dataset}')
        # print(f'saving alpha_calibrated_attribute at log/alpha_calibrated_attribute_knowledge_pc_{args.dataset}')