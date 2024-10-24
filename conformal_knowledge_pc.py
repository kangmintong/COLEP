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

def conformal_knowledge_pc(args):
    print(f'Loading dataset')
    if args.dataset == 'GTSRB':
        data = DataMain(batch_size=args.batch_size)
        data.data_set_up(istrain=False)
        data.greeting()
        num_label, num_attribute, num_channel = 12, 13, 3

        mappings_label, mappings_attribute, rule_weight = get_knowledge(args.dataset, use_pc=True)

        print(f'Loading models')
        if args.load_exist:
            print(f'loading the model: log/mdoel_pc_reasonsing_{args.dataset}_{args.sigma}')
            model_pc_reasoning = torch.load(f'log/mdoel_pc_reasonsing_{args.dataset}_{args.sigma}')
        else:
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

        if args.load_exist:
            print(f'loading the model: log/mdoel_pc_reasonsing_{args.dataset}_{args.sigma}')
            model_pc_reasoning = torch.load(f'log/mdoel_pc_reasonsing_{args.dataset}_{args.sigma}')
        else:
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
    # args.alpha = args.alpha / (num_label + num_attribute)

    if args.method_conformal == 'split_conformal':
        if args.calibrate and not args.inference:
            alpha_calibrated_label = []
            alpha_calibrated_attribute = []

            y_hat_all = torch.zeros((len(data.y_val), num_label+num_attribute, 2)).cuda()
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

            print(f'Accuracy: {(torch.argmax(y_hat_all[:,:num_label,1].detach().cpu(), dim=1) == data.y_val).sum() / len(data.y_val)}')

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
            torch.save(alpha_calibrated_label, 'log/alpha_calibrated_label_knowledge_pc')
            print(f'alpha_calibrated_attribute: {alpha_calibrated_attribute}')
            torch.save(alpha_calibrated_attribute, 'log/alpha_calibrated_attribute_knowledge_pc')
            print('saving alpha_calibrated_label at log/alpha_calibrated_label_knowledge_pc')
            print('saving alpha_calibrated_attribute at log/alpha_calibrated_attribute_knowledge_pc')
        elif not args.calibrate and args.inference:
            alpha_calibrated_label = torch.load('log/alpha_calibrated_label_knowledge_pc')
            alpha_calibrated_attribute = torch.load('log/alpha_calibrated_attribute_knowledge_pc')

            if args.attack_type=='physical_attack':
                data.X_test = np.load('./data/physical_attack/stop_sign_adv_X_test.npy')
                data.y_test = np.load('./data/physical_attack/stop_sign_label.npy')
                data.X_test = np.array([data.pre_process_image(data.X_test[i]) for i in range(len(data.X_test))],dtype=np.float32)
                data.y_test = np.array(data.y_test, dtype=np.long)
                data.X_test = torch.FloatTensor(data.X_test)
                data.X_test = data.X_test.permute(0, 3, 1, 2)
                data.y_test = torch.LongTensor(data.y_test)

            if args.attack_type=='pgd':
                print('Performing pgd attack!')
                print('Using BPDA attack since the PC component is non-differentiable!')
                model_pc_reasoning.args.pc_correction = 0
                X, GT = data.sequential_test_batch()
                cnt = 0
                while X is not None:
                    cnt += 1
                    print(f'pgd attack on batch {cnt}')
                    X = X.cuda()
                    GT_attr = torch.from_numpy(mappings_attribute[:, GT.numpy()]).transpose(0, 1)
                    GT = GT.cuda()
                    GT_attr = GT_attr.cuda()
                    X = conformal_attack_pc(X, model_pc_reasoning, GT, GT_attr, max_norm=args.max_norm, num_class=num_label)
                    torch.save(X, f'log/adv_sample/adv_knowledge_PC_{cnt}')
                    X, GT = data.sequential_test_batch()
                model_pc_reasoning.args.pc_correction = 1

            y_hat_all = torch.zeros((len(data.y_test), num_label+num_attribute, 2)).cuda()
            X, GT = data.sequential_test_batch()
            cur = 0
            cnt = 0
            with torch.no_grad():
                while X is not None:
                    cnt += 1
                    X = X.cuda()
                    if args.attack_type=='pgd':
                        X = torch.load(f'log/adv_sample/adv_knowledge_PC_{cnt}')
                    Y = model_pc_reasoning(X)
                    this_batch_size = len(Y)
                    y_hat_all[cur:cur + this_batch_size, :, :] = Y
                    cur = cur + this_batch_size
                    X, GT = data.sequential_test_batch()

            size_all = torch.zeros((len(data.y_test)))
            size_all = size_all + num_label
            label_test = data.y_test
            y_hat = {}
            for i in range(len(label_test)):
                y_hat[i] = list(range(0, num_label))

            data = DataMain(batch_size=args.batch_size)
            data.data_set_up(istrain=False)

            # conformal inference on label sensors
            for i in tqdm(range(num_label)):
                P_test = y_hat_all[:, i, :]
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
            print(f'Use knowledge_set_correction: {args.knowledge_set_correction}')
            for i in tqdm(range(num_attribute)):
                P_test = y_hat_all[:, i + num_label, :]
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


# import argparse
# from used_knowledge.used_knowledge import get_knowledge
# import numpy as np
# import pandas as pd
# import os.path
# from os import path
# from dataset.dataset_conformal import GetDataset
# import random
# import torch
# import sys
# from arc.classification import ProbabilityAccumulator as ProbAccum
# from dataset.dataset import DataMain
# from model.model import NEURAL
# from model.model_single import NEURAL_single
# from tqdm import tqdm
# from scipy.stats.mstats import mquantiles
# import arc
# from conformal_attack import conformal_attack
# from conformal_attack import conformal_attack_knowledge
# from knowledge_probabilistic_circuit.knwoledge_pc import knowledge_pc
# from conformal_attack import conformal_attack_pc
# from dataset.dataset import Cifar10
# from model.resnet import resnet56
#
# def conformal_knowledge_pc(args):
#     print(f'Loading dataset')
#     if args.dataset == 'GTSRB':
#         data = DataMain(batch_size=args.batch_size)
#         data.data_set_up(istrain=False)
#         data.greeting()
#         num_label, num_attribute, num_channel = 12, 13, 3
#
#         mappings_label, mappings_attribute, rule_weight = get_knowledge(args.dataset, use_pc=True)
#
#         print(f'Loading models')
#         model_list_label = []
#         model_list_attribute = []
#         for i in range(num_label):
#             model = NEURAL(n_class=1, n_channel=num_channel)
#             if os.path.exists("pretrained_models/hier/model_%d_%.2f_5.pt" % (i, args.sigma)):
#                 model.load_state_dict(torch.load("pretrained_models/hier/model_%d_%.2f_5.pt" % (i, args.sigma)))
#             else:
#                 model.load_state_dict(torch.load("pretrained_models/hier/model_%d_%.2f.pt" % (i, args.sigma)))
#             model = model.cuda()
#             model.eval()
#             model_list_label.append(model)
#         for i in range(num_attribute):
#             model = NEURAL(n_class=1, n_channel=num_channel)
#             if os.path.exists("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, args.sigma)):
#                 model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, args.sigma)))
#             else:
#                 model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f.pt" % (i, args.sigma)))
#             model = model.cuda()
#             model.eval()
#             model_list_attribute.append(model)
#
#         knowledge = np.transpose(mappings_attribute)
#         rule_weight = np.transpose(rule_weight)
#         knowledge = torch.from_numpy(knowledge).cuda()
#         knowledge_weight = torch.zeros_like(knowledge)
#         for i in range(knowledge.shape[0]):
#             for j in range(knowledge.shape[1]):
#                 if knowledge[i][j] == 1:
#                     knowledge_weight[i][j] = rule_weight[i][j] * args.knowledge_weights
#         model_pc_reasoning = knowledge_pc(model_list_label, model_list_attribute, knowledge, knowledge_weight, args)
#     elif args.dataset == 'cifar10':
#         data = Cifar10(batch_size=args.batch_size)
#         data.data_set_up(istrain=False)
#         data.greeting()
#         num_label, num_attribute, num_channel = 10, 9, 3
#
#         mappings_label, mappings_attribute, rule_weight = get_knowledge(args.dataset, use_pc=True)
#
#         print(f'Loading models')
#         model_list_label = []
#         model_list_attribute = []
#         for i in range(num_label):
#             model = resnet56(n_class=1)
#             model = torch.load(f'pretrained_models/cifar10/model_sensor_{i}_cifar10_{args.sigma}')
#             model = model.cuda()
#             model.eval()
#             model_list_label.append(model)
#         for i in range(num_attribute):
#             model = resnet56(n_class=1)
#             model = torch.load(f'pretrained_models/cifar10/model_sensor_{i+10}_cifar10_{args.sigma}')
#             model = model.cuda()
#             model.eval()
#             model_list_attribute.append(model)
#
#         knowledge = np.transpose(mappings_attribute)
#         rule_weight = np.transpose(rule_weight)
#         knowledge = torch.from_numpy(knowledge).cuda()
#         knowledge_weight = torch.zeros_like(knowledge)
#         for i in range(knowledge.shape[0]):
#             for j in range(knowledge.shape[1]):
#                 if knowledge[i][j] == 1:
#                     knowledge_weight[i][j] = rule_weight[i][j] * args.knowledge_weights
#         model_pc_reasoning = knowledge_pc(model_list_label, model_list_attribute, knowledge, knowledge_weight, args)
#     else:
#         print(f'Dataset {args.dataset} is not implemented!')
#         sys.exit(1)
#
#     # Unioun bound
#     args.alpha = args.alpha / (num_label + num_attribute)
#
#     if args.method_conformal == 'split_conformal':
#         if args.calibrate and not args.inference:
#             alpha_calibrated_label = []
#             alpha_calibrated_attribute = []
#
#             y_hat_all = torch.zeros((len(data.y_val), num_label+num_attribute, 2)).cuda()
#             X, GT = data.sequential_val_batch()
#             cur = 0
#             with torch.no_grad():
#                 while X is not None:
#                     X = X.cuda()
#                     Y = model_pc_reasoning(X)
#                     # model_pc_reasoning.args.pc_correction=0
#                     # Y_ = model_pc_reasoning(X)
#                     # print(Y-Y_)
#                     # model_pc_reasoning.args.pc_correction = 1
#                     this_batch_size = len(Y)
#                     y_hat_all[cur:cur + this_batch_size, :, :] = Y.sigmoid()
#                     cur = cur + this_batch_size
#                     X, GT = data.sequential_val_batch()
#
#             print(f'Accuracy: {(torch.argmax(y_hat_all[:,:num_label,1].detach().cpu(), dim=1) == data.y_val).sum() / len(data.y_val)}')
#
#             # calibrate label sensors
#             for i in tqdm(range(num_label)):
#                 label_mapping = mappings_label[i]
#                 cur = 0
#                 y_hat = y_hat_all[:, i]
#                 label = torch.zeros((len(data.y_val))).cuda()
#                 X, GT = data.sequential_val_batch()
#                 while X is not None:
#                     X = X.cuda()
#                     GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
#                     this_batch_size = len(X)
#                     label[cur:cur + this_batch_size] = GT
#                     cur = cur + this_batch_size
#                     X, GT = data.sequential_val_batch()
#                 y_hat = y_hat.detach().cpu()
#                 label = label.cpu()
#
#                 n2 = y_hat.shape[0]
#                 grey_box = ProbAccum(y_hat)
#                 rng = np.random.default_rng(args.seed)
#                 epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
#                 label = label.int()
#                 alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
#
#                 scores = args.alpha - alpha_max
#                 level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
#                 alpha_correction = mquantiles(scores, prob=level_adjusted)
#                 alpha_calibrated_ = args.alpha - alpha_correction
#                 alpha_calibrated_label.append(alpha_calibrated_[0])
#
#             # calibrate attribute sensors
#             for i in tqdm(range(num_attribute)):
#                 label_mapping = mappings_attribute[i]
#                 cur = 0
#                 y_hat = y_hat_all[:, i + num_label]
#                 label = torch.zeros((len(data.y_val))).cuda()
#
#                 X, GT = data.sequential_val_batch()
#                 while X is not None:
#                     X = X.cuda()  # to(device)
#                     GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
#                     this_batch_size = len(X)
#                     label[cur:cur + this_batch_size] = GT
#                     cur = cur + this_batch_size
#                     X, GT = data.sequential_val_batch()
#                 y_hat = y_hat.detach().cpu()
#                 label = label.cpu()
#
#                 n2 = y_hat.shape[0]
#                 grey_box = ProbAccum(y_hat)
#                 rng = np.random.default_rng(args.seed)
#                 epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
#                 label = label.int()
#                 alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
#
#                 scores = args.alpha - alpha_max
#                 level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
#                 alpha_correction = mquantiles(scores, prob=level_adjusted)
#                 alpha_calibrated_ = args.alpha - alpha_correction
#                 alpha_calibrated_attribute.append(alpha_calibrated_[0])
#
#
#             print(f'alpha_calibrated_label: {alpha_calibrated_label}')
#             torch.save(alpha_calibrated_label, f'log/alpha_calibrated_label_knowledge_pc_{args.dataset}')
#             print(f'alpha_calibrated_attribute: {alpha_calibrated_attribute}')
#             torch.save(alpha_calibrated_attribute, f'log/alpha_calibrated_attribute_knowledge_pc_{args.dataset}')
#             print(f'saving alpha_calibrated_label at log/alpha_calibrated_label_knowledge_pc_{args.dataset}')
#             print(f'saving alpha_calibrated_attribute at log/alpha_calibrated_attribute_knowledge_pc_{args.dataset}')
#         elif not args.calibrate and args.inference:
#             alpha_calibrated_label = torch.load(f'log/alpha_calibrated_label_knowledge_pc_{args.dataset}')
#             alpha_calibrated_attribute = torch.load(f'log/alpha_calibrated_attribute_knowledge_pc_{args.dataset}')
#
#             if args.attack_type=='physical_attack':
#                 data.X_test = np.load('./data/physical_attack/stop_sign_adv_X_test.npy')
#                 data.y_test = np.load('./data/physical_attack/stop_sign_label.npy')
#                 data.X_test = np.array([data.pre_process_image(data.X_test[i]) for i in range(len(data.X_test))],dtype=np.float32)
#                 data.y_test = np.array(data.y_test, dtype=np.long)
#                 data.X_test = torch.FloatTensor(data.X_test)
#                 data.X_test = data.X_test.permute(0, 3, 1, 2)
#                 data.y_test = torch.LongTensor(data.y_test)
#
#             if args.attack_type=='pgd' or args.attack_type=='smoothadv':
#                 print('Performing pgd attack!')
#                 print('Using BPDA attack since the PC component is non-differentiable!')
#                 model_pc_reasoning.args.pc_correction = 0
#                 X, GT = data.sequential_test_batch()
#                 cnt = 0
#                 while X is not None:
#                     cnt += 1
#                     print(f'pgd attack on batch {cnt}')
#                     X = X.cuda()
#                     GT_attr = torch.from_numpy(mappings_attribute[:, GT.numpy()]).transpose(0, 1)
#                     GT = GT.cuda()
#                     GT_attr = GT_attr.cuda()
#                     if args.attack_type=='smoothadv':
#                         X = conformal_attack_pc(X, model_pc_reasoning, GT, GT_attr, max_norm=args.max_norm,num_class=num_label,smoothadv=True)
#                     else:
#                         X = conformal_attack_pc(X, model_pc_reasoning, GT, GT_attr, max_norm=args.max_norm,
#                                                 num_class=num_label, smoothadv=False)
#                     torch.save(X, f'log/adv_sample/adv_knowledge_PC_{cnt}_{args.dataset}')
#                     X, GT = data.sequential_test_batch()
#                 model_pc_reasoning.args.pc_correction = 1
#
#             y_hat_all = torch.zeros((len(data.y_test), num_label+num_attribute, 2)).cuda()
#             X, GT = data.sequential_test_batch()
#             cur = 0
#             cnt = 0
#             with torch.no_grad():
#                 while X is not None:
#                     cnt += 1
#                     X = X.cuda()
#                     if args.attack_type=='pgd' or args.attack_type=='smoothadv':
#                         X = torch.load(f'log/adv_sample/adv_knowledge_PC_{cnt}_{args.dataset}')
#                     Y = model_pc_reasoning(X)
#                     this_batch_size = len(Y)
#                     y_hat_all[cur:cur + this_batch_size, :, :] = Y.sigmoid()
#                     cur = cur + this_batch_size
#                     X, GT = data.sequential_test_batch()
#
#             size_all = torch.zeros((len(data.y_test)))
#             size_all = size_all + num_label
#             label_test = data.y_test
#             y_hat = {}
#             for i in range(len(label_test)):
#                 y_hat[i] = list(range(0, num_label))
#
#             data = DataMain(batch_size=args.batch_size)
#             data.data_set_up(istrain=False)
#
#             # conformal inference on label sensors
#             for i in tqdm(range(num_label)):
#                 P_test = y_hat_all[:, i, :]
#                 P_test = P_test.detach().cpu()
#
#                 rng = np.random.default_rng(args.seed)
#                 epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
#                 grey_box_test = ProbAccum(P_test)
#                 S_hat = grey_box_test.predict_sets(alpha_calibrated_label[i], epsilon=epsilon, allow_empty=False)
#
#                 for k, l in enumerate(S_hat):
#                     # if size_all[k] == 1:
#                     #     continue
#                     if len(l) == 1 and l[0] == 0:
#                         size_all[k] -= 1
#                         y_hat[k].remove(i)
#                     # elif len(l) == 1 and l[0] == 1:
#                     #     size_all[k] = 1
#                     #     y_hat[k] = [i]
#
#             # conformal inference on attribute sensors
#             print(f'Use knowledge_set_correction: {args.knowledge_set_correction}')
#             for i in tqdm(range(num_attribute)):
#                 P_test = y_hat_all[:, i + num_label, :]
#                 P_test = P_test.detach().cpu()
#
#                 rng = np.random.default_rng(args.seed)
#                 epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
#                 grey_box_test = ProbAccum(P_test)
#                 S_hat = grey_box_test.predict_sets(alpha_calibrated_attribute[i], epsilon=epsilon, allow_empty=False)
#
#                 if args.knowledge_set_correction:
#                     for k, l in enumerate(S_hat):
#                         # if size_all[k] == 1:
#                         #     continue
#                         if len(l) == 1 and l[0] == 0:
#                             for jj in range(len(mappings_attribute[i])):
#                                 if mappings_attribute[i][jj] == 1:
#                                     if jj in y_hat[k]:
#                                         size_all[k] -= 1
#                                         y_hat[k].remove(jj)
#                         elif len(l) == 1 and l[0] == 1:
#                             for jj in range(len(mappings_attribute[i])):
#                                 if mappings_attribute[i][jj] == 0:
#                                     if jj in y_hat[k]:
#                                         size_all[k] -= 1
#                                         y_hat[k].remove(jj)
#             # evaluation
#             avg_size = size_all.sum() / len(size_all)
#             marginal_coverage = 0
#             for i in range(len(label_test)):
#                 if label_test[i] in y_hat[i]:
#                     marginal_coverage += 1
#             marginal_coverage = marginal_coverage / len(label_test)
#             print(f'marginal coverage under attack {args.attack_type}: {marginal_coverage}')
#             print(f'avg_size under attack {args.attack_type}: {avg_size}')
#         else:
#             print(f'args.calibrate={args.calibrate}, args.inference={args.inference} are not satisfactory!')
#             sys.exit(1)
#     else:
#         print(f'Conformal method {args.method_conformal} is not implemented!')
#         sys.exit(1)
#
#
#
# # import argparse
# # from used_knowledge.used_knowledge import get_knowledge
# # import numpy as np
# # import pandas as pd
# # import os.path
# # from os import path
# # from dataset.dataset_conformal import GetDataset
# # import random
# # import torch
# # import sys
# # from arc.classification import ProbabilityAccumulator as ProbAccum
# # from dataset.dataset import DataMain
# # from model.model import NEURAL
# # from model.model_single import NEURAL_single
# # from tqdm import tqdm
# # from scipy.stats.mstats import mquantiles
# # import arc
# # from conformal_attack import conformal_attack
# # from conformal_attack import conformal_attack_knowledge
# # from knowledge_probabilistic_circuit.knwoledge_pc import knowledge_pc
# # from conformal_attack import conformal_attack_pc
# # from dataset.dataset import Cifar10
# # from model.resnet import resnet56
# # import time
# # from statsmodels.stats.proportion import proportion_confint
# # from scipy.stats import norm
# # import math
# #
# # def conformal_knowledge_pc(args):
# #     print(f'Loading dataset')
# #     if args.dataset == 'GTSRB':
# #         data = DataMain(batch_size=args.batch_size)
# #         data.data_set_up(istrain=False)
# #         data.greeting()
# #         num_label, num_attribute, num_channel = 12, 13, 3
# #
# #         mappings_label, mappings_attribute, rule_weight = get_knowledge(args.dataset, use_pc=True)
# #
# #         print(f'Loading models')
# #         model_list_label = []
# #         model_list_attribute = []
# #         for i in range(num_label):
# #             model = NEURAL(n_class=1, n_channel=num_channel)
# #             if os.path.exists("pretrained_models/hier/model_%d_%.2f_5.pt" % (i, args.sigma)):
# #                 model.load_state_dict(torch.load("pretrained_models/hier/model_%d_%.2f_5.pt" % (i, args.sigma)))
# #             else:
# #                 model.load_state_dict(torch.load("pretrained_models/hier/model_%d_%.2f.pt" % (i, args.sigma)))
# #             model = model.cuda()
# #             model.eval()
# #             model_list_label.append(model)
# #         for i in range(num_attribute):
# #             model = NEURAL(n_class=1, n_channel=num_channel)
# #             if os.path.exists("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, args.sigma)):
# #                 model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, args.sigma)))
# #             else:
# #                 model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f.pt" % (i, args.sigma)))
# #             model = model.cuda()
# #             model.eval()
# #             model_list_attribute.append(model)
# #
# #         knowledge = np.transpose(mappings_attribute)
# #         rule_weight = np.transpose(rule_weight)
# #         knowledge = torch.from_numpy(knowledge).cuda()
# #         knowledge_weight = torch.zeros_like(knowledge)
# #         for i in range(knowledge.shape[0]):
# #             for j in range(knowledge.shape[1]):
# #                 if knowledge[i][j] == 1:
# #                     knowledge_weight[i][j] = rule_weight[i][j] * args.knowledge_weights
# #         model_pc_reasoning = knowledge_pc(model_list_label, model_list_attribute, knowledge, knowledge_weight, args)
# #     elif args.dataset == 'cifar10':
# #         data = Cifar10(batch_size=args.batch_size)
# #         data.data_set_up(istrain=False)
# #         data.greeting()
# #         num_label, num_attribute, num_channel = 10, 9, 3
# #
# #         mappings_label, mappings_attribute, rule_weight = get_knowledge(args.dataset, use_pc=True)
# #
# #         print(f'Loading models')
# #         model_list_label = []
# #         model_list_attribute = []
# #         for i in range(num_label):
# #             model = resnet56(n_class=1)
# #             model = torch.load(f'pretrained_models/cifar10/model_sensor_{i}_cifar10_{args.sigma}')
# #             model = model.cuda()
# #             model.eval()
# #             model_list_label.append(model)
# #         for i in range(num_attribute):
# #             model = resnet56(n_class=1)
# #             model = torch.load(f'pretrained_models/cifar10/model_sensor_{i+10}_cifar10_{args.sigma}')
# #             model = model.cuda()
# #             model.eval()
# #             model_list_attribute.append(model)
# #
# #         knowledge = np.transpose(mappings_attribute)
# #         rule_weight = np.transpose(rule_weight)
# #         knowledge = torch.from_numpy(knowledge).cuda()
# #         knowledge_weight = torch.zeros_like(knowledge)
# #         for i in range(knowledge.shape[0]):
# #             for j in range(knowledge.shape[1]):
# #                 if knowledge[i][j] == 1:
# #                     knowledge_weight[i][j] = rule_weight[i][j] * args.knowledge_weights
# #         model_pc_reasoning = knowledge_pc(model_list_label, model_list_attribute, knowledge, knowledge_weight, args)
# #     else:
# #         print(f'Dataset {args.dataset} is not implemented!')
# #         sys.exit(1)
# #
# #     # Unioun bound
# #     # args.alpha = args.alpha / (num_label + num_attribute)
# #
# #     if args.method_conformal == 'split_conformal':
# #         if args.calibrate and not args.inference:
# #             alpha_calibrated_label = []
# #             alpha_calibrated_attribute = []
# #
# #             data = DataMain(batch_size=200)
# #             data.data_set_up(istrain=False)
# #             y_hat_all = torch.zeros((len(data.y_val), num_label+num_attribute, 2)).cuda()
# #             label = data.y_val
# #             X, GT = data.sequential_val_batch()
# #             cur = 0
# #             with torch.no_grad():
# #                 while X is not None:
# #                     X = X.cuda()
# #                     Y = model_pc_reasoning(X)
# #                     # model_pc_reasoning.args.pc_correction=0
# #                     # Y_ = model_pc_reasoning(X)
# #                     # print(Y-Y_)
# #                     # model_pc_reasoning.args.pc_correction = 1
# #                     this_batch_size = len(Y)
# #                     y_hat_all[cur:cur + this_batch_size, :, :] = Y
# #                     cur = cur + this_batch_size
# #                     X, GT = data.sequential_val_batch()
# #
# #             print(f'Accuracy: {(torch.argmax(y_hat_all[:,:num_label,1].detach().cpu(), dim=1) == data.y_val).sum() / len(data.y_val)}')
# #             y_hat_all = y_hat_all[:,:num_label,1].softmax(-1).detach().cpu()
# #
# #             data = DataMain(batch_size=args.batch_size)
# #             data.data_set_up(istrain=False)
# #
# #             # randomized smoothing for each sensor
# #             delta = args.max_norm / args.sigma_certify
# #
# #             delta = 0.
# #
# #             label_test = []
# #             start_time = time.time()
# #             with torch.no_grad():
# #                 X, GT = data.sequential_val_batch()
# #                 cnt = 0
# #                 num_certified = 0
# #                 pAs = np.zeros((args.num_certify, num_label + num_attribute, 2))
# #                 sensor_upbnd = np.zeros((args.num_certify, num_label + num_attribute))
# #                 sensor_lowbnd = np.zeros((args.num_certify, num_label + num_attribute))
# #                 while X is not None:
# #                     cnt += 1
# #                     if cnt % args.skip_certify and num_certified < args.num_certify:
# #                         X = X.cuda()
# #                         label_test.append(GT)
# #                         for i in range(num_label):
# #                             if i<num_label:
# #                                 model = model_list_label[i]
# #                             else:
# #                                 model = model_list_attribute[i-num_label]
# #
# #                             # counts = model_pc_reasoning.smoothed_forward(model, X, args.sigma_certify, N=args.N_certify, pc_correction=args.pc_correction)
# #                             # pos = counts.argmax(0)
# #                             # pa = proportion_confint(counts[pos].item(), args.N_certify, alpha=2 * args.alpha_certify, method="beta")
# #                             N = args.N_certify
# #                             batch_size = 5000
# #                             with torch.no_grad():
# #                                 pred = torch.zeros(2).cuda()
# #                                 for _ in range(math.ceil(N / batch_size)):
# #                                     this_batch_size = min(batch_size, N)
# #                                     N -= this_batch_size
# #                                     batch = X.repeat((this_batch_size, 1, 1, 1))
# #                                     noise = torch.randn_like(batch, device='cuda') * args.sigma_certify * 0.
# #                                     predictions = model(batch + noise)[:, :]
# #                                     predictions = predictions.softmax(-1)
# #                                     pred = pred + predictions.mean(0)
# #                                 # print(f'pred before: {pred}')
# #                                 # print(f'{math.ceil(args.N_certify / batch_size)}')
# #                                 pred = pred / math.ceil(args.N_certify / batch_size)
# #                                 # print(pred)
# #
# #
# #                             pred = pred.cpu()
# #                             sensor_upbnd[num_certified,i] = norm.cdf(norm.ppf(pred[1]) + delta)
# #                             sensor_lowbnd[num_certified,i] = norm.cdf(norm.ppf(pred[1]) - delta)
# #                             # else:
# #                             #     sensor_upbnd[num_certified,i] = norm.cdf(norm.ppf(1-pred[0]) + delta)
# #                             #     sensor_lowbnd[num_certified,i] = norm.cdf(norm.ppf(1-pred) - delta)
# #                             # print(f'pa: {pa}')
# #                             # print(f'norm.ppf(pa[1]): {norm.ppf(pa[1])}')
# #                             # print(f'delta: {delta}')
# #                             # print(f'norm.cdf(norm.ppf(pa[1]) + delta): {norm.cdf(norm.ppf(pa[1]) + delta)} \n')
# #                         num_certified += 1
# #                         cur_time = time.time()
# #                         print(f'certify {num_certified} samples, wall clock time {cur_time-start_time}, bound interval avg: {(sensor_upbnd[num_certified-1,:]-sensor_lowbnd[num_certified-1,:]).mean()}')
# #                     X, GT = data.sequential_val_batch()
# #                 sensor_upbnd[sensor_upbnd>1.0] = 1.0
# #                 sensor_lowbnd[sensor_lowbnd<0.0] = 0.0
# #
# #             print(f'mean bound interval before PC correction: {(sensor_upbnd - sensor_lowbnd).mean()}')
# #             # certification for the PC part
# #             start_time_certify = time.time()
# #             sensor_lowbnd_pc_correction, sensor_upbnd_pc_correction = model_pc_reasoning.certify_PC(sensor_lowbnd[:,:num_label], sensor_upbnd[:,:num_label], sensor_lowbnd[:,num_label:], sensor_upbnd[:,num_label:], pc_correction=args.pc_correction)
# #             end_time_certify = time.time()
# #             print(f'certification time of {len(sensor_lowbnd_pc_correction)} samples: {end_time_certify-start_time_certify}')
# #
# #             print(f'mean bound interval after PC correction: {(sensor_upbnd_pc_correction-sensor_lowbnd_pc_correction).mean()}')
# #
# #             # construct certified robust prediction set
# #             certified_prediction_set = {}
# #             for i in range(len(sensor_lowbnd_pc_correction)):
# #                 certified_prediction_set[i] = list(range(0, num_label))
# #             y_hat_all = np.zeros((num_certified, num_label + num_attribute, 2))
# #
# #             for i in range(num_certified):
# #                 label_mapping = mappings_label[i]
# #                 cur_label = torch.from_numpy(label_mapping[np.array([label_test[i]])])
# #                 print(f'{sensor_lowbnd_pc_correction[i, 0]} -- {sensor_upbnd_pc_correction[i, 0]}')
# #                 for j in range(num_label + num_attribute):
# #                     if cur_label==1:
# #                         y_hat_all[i, j, 1] = sensor_lowbnd_pc_correction[i, j]
# #                         y_hat_all[i, j, 0] = 1 - sensor_lowbnd_pc_correction[i, j]
# #                     elif cur_label==0:
# #                         y_hat_all[i, j, 1] = sensor_upbnd_pc_correction[i, j]
# #                         y_hat_all[i, j, 0] = 1 - sensor_upbnd_pc_correction[i, j]
# #                     else:
# #                         y_hat_all[i, j, 0] = y_hat_all[i, j, 1] = 0.5
# #
# #             # conformal calibration
# #             y_hat_all = y_hat_all[:,:,1]
# #             for i in range(y_hat_all.shape[0]):
# #                 y_hat_all[i] = np.exp(y_hat_all[i])
# #                 sum_ = y_hat_all[i].sum()
# #                 y_hat_all[i] /= sum_
# #             label = np.array(label_test)
# #
# #
# #             y_hat = y_hat_all
# #             n2 = y_hat.shape[0]
# #             if args.score_type == 'aps':
# #                 grey_box = ProbAccum(y_hat)
# #                 rng = np.random.default_rng(args.seed)
# #                 epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
# #                 alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
# #                 scores = args.alpha - alpha_max
# #                 level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
# #                 alpha_correction = mquantiles(scores, prob=level_adjusted)
# #                 alpha_calibrated_label = args.alpha - alpha_correction
# #             elif args.score_type == 'hps':
# #                 scores = torch.zeros(n2)
# #                 for i in range(n2):
# #                     scores[i] = 1 - y_hat[i][label[i]]
# #                 # scores = args.alpha - scores
# #                 level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
# #                 alpha_correction = mquantiles(scores, prob=level_adjusted)
# #                 alpha_calibrated_label = alpha_correction
# #
# #             # calibrate label sensors
# #             # for i in tqdm(range(num_label)):
# #             #     label_mapping = mappings_label[i]
# #             #     cur = 0
# #             #     y_hat = y_hat_all[:, i]
# #             #     label = torch.from_numpy(label_mapping[label_test])
# #             #     # X, GT = data.sequential_val_batch()
# #             #     # while X is not None:
# #             #     #     X = X.cuda()
# #             #     #     GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
# #             #     #     this_batch_size = len(X)
# #             #     #     label[cur:cur + this_batch_size] = GT
# #             #     #     cur = cur + this_batch_size
# #             #     #     X, GT = data.sequential_val_batch()
# #             #     # y_hat = y_hat
# #             #     # label = label.cpu()
# #             #
# #             #     n2 = y_hat.shape[0]
# #             #     grey_box = ProbAccum(y_hat)
# #             #     rng = np.random.default_rng(args.seed)
# #             #     epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
# #             #     label = label.int()
# #             #
# #             #     # print(f'n2: {}')
# #             #     alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
# #             #
# #             #     scores = args.alpha - alpha_max
# #             #     level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
# #             #     alpha_correction = mquantiles(scores, prob=level_adjusted)
# #             #     alpha_calibrated_ = args.alpha - alpha_correction
# #             #     alpha_calibrated_label.append(alpha_calibrated_[0])
# #
# #             # calibrate attribute sensors
# #             # for i in tqdm(range(num_attribute)):
# #             #     label_mapping = mappings_attribute[i]
# #             #     cur = 0
# #             #     y_hat = y_hat_all[:, i + num_label]
# #             #     label = torch.zeros((len(data.y_val))).cuda()
# #             #
# #             #     X, GT = data.sequential_val_batch()
# #             #     while X is not None:
# #             #         X = X.cuda()  # to(device)
# #             #         GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
# #             #         this_batch_size = len(X)
# #             #         label[cur:cur + this_batch_size] = GT
# #             #         cur = cur + this_batch_size
# #             #         X, GT = data.sequential_val_batch()
# #             #     y_hat = y_hat.detach().cpu()
# #             #     label = label.cpu()
# #             #
# #             #     n2 = y_hat.shape[0]
# #             #     grey_box = ProbAccum(y_hat)
# #             #     rng = np.random.default_rng(args.seed)
# #             #     epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
# #             #     label = label.int()
# #             #     alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
# #             #
# #             #     scores = args.alpha - alpha_max
# #             #     level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
# #             #     alpha_correction = mquantiles(scores, prob=level_adjusted)
# #             #     alpha_calibrated_ = args.alpha - alpha_correction
# #             #     alpha_calibrated_attribute.append(alpha_calibrated_[0])
# #
# #             alpha_calibrated_label = alpha_calibrated_label
# #             print(f'alpha_calibrated_label: {alpha_calibrated_label}')
# #             torch.save(alpha_calibrated_label, f'log/alpha_calibrated_label_knowledge_pc_{args.dataset}')
# #             print(f'alpha_calibrated_attribute: {alpha_calibrated_attribute}')
# #             torch.save(alpha_calibrated_attribute, f'log/alpha_calibrated_attribute_knowledge_pc_{args.dataset}')
# #             print(f'saving alpha_calibrated_label at log/alpha_calibrated_label_knowledge_pc_{args.dataset}')
# #             print(f'saving alpha_calibrated_attribute at log/alpha_calibrated_attribute_knowledge_pc_{args.dataset}')
# #         elif not args.calibrate and args.inference:
# #             alpha_calibrated_label = torch.load(f'log/alpha_calibrated_label_knowledge_pc_{args.dataset}')
# #             alpha_calibrated_attribute = torch.load(f'log/alpha_calibrated_attribute_knowledge_pc_{args.dataset}')
# #
# #             if args.attack_type=='physical_attack':
# #                 data.X_test = np.load('./data/physical_attack/stop_sign_adv_X_test.npy')
# #                 data.y_test = np.load('./data/physical_attack/stop_sign_label.npy')
# #                 data.X_test = np.array([data.pre_process_image(data.X_test[i]) for i in range(len(data.X_test))],dtype=np.float32)
# #                 data.y_test = np.array(data.y_test, dtype=np.long)
# #                 data.X_test = torch.FloatTensor(data.X_test)
# #                 data.X_test = data.X_test.permute(0, 3, 1, 2)
# #                 data.y_test = torch.LongTensor(data.y_test)
# #
# #             if args.attack_type=='pgd' or args.attack_type=='smoothadv':
# #                 print('Performing pgd attack!')
# #                 print('Using BPDA attack since the PC component is non-differentiable!')
# #                 model_pc_reasoning.args.pc_correction = 0
# #                 X, GT = data.sequential_test_batch()
# #                 cnt = 0
# #                 while X is not None:
# #                     cnt += 1
# #                     print(f'pgd attack on batch {cnt}')
# #                     X = X.cuda()
# #                     GT_attr = torch.from_numpy(mappings_attribute[:, GT.numpy()]).transpose(0, 1)
# #                     GT = GT.cuda()
# #                     GT_attr = GT_attr.cuda()
# #                     X = conformal_attack_pc(X, model_pc_reasoning, GT, GT_attr, max_norm=args.max_norm,num_class=num_label,smoothadv=True)
# #                     torch.save(X, f'log/adv_sample/adv_knowledge_PC_{cnt}_{args.dataset}')
# #                     X, GT = data.sequential_test_batch()
# #                 model_pc_reasoning.args.pc_correction = 1
# #
# #             y_hat_all = torch.zeros((len(data.y_test), num_label+num_attribute, 2)).cuda()
# #             X, GT = data.sequential_test_batch()
# #             cur = 0
# #             cnt = 0
# #             with torch.no_grad():
# #                 while X is not None:
# #                     cnt += 1
# #                     X = X.cuda()
# #                     if args.attack_type=='pgd' or args.attack_type=='smoothadv':
# #                         X = torch.load(f'log/adv_sample/adv_knowledge_PC_{cnt}_{args.dataset}')
# #                     Y = model_pc_reasoning(X)
# #                     this_batch_size = len(Y)
# #                     y_hat_all[cur:cur + this_batch_size, :, :] = Y
# #                     cur = cur + this_batch_size
# #                     X, GT = data.sequential_test_batch()
# #
# #             size_all = torch.zeros((len(data.y_test)))
# #             size_all = size_all + num_label
# #             label_test = data.y_test
# #             y_hat = {}
# #             for i in range(len(label_test)):
# #                 y_hat[i] = list(range(0, num_label))
# #
# #             data = DataMain(batch_size=args.batch_size)
# #             data.data_set_up(istrain=False)
# #
# #             P_test = y_hat_all[:,:num_label,1].softmax(-1).cpu()
# #
# #
# #             if args.score_type == 'aps':
# #                 rng = np.random.default_rng(args.seed)
# #                 epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
# #                 grey_box_test = ProbAccum(P_test)
# #                 S_hat = grey_box_test.predict_sets(alpha_calibrated_label, epsilon=epsilon, allow_empty=False)
# #             elif args.score_type == 'hps':
# #                 S_hat = []
# #                 n2 = len(P_test)
# #                 for i in range(n2):
# #                     tmp = []
# #                     for j in range(num_label):
# #                         if 1 - P_test[i][j].item() <= alpha_calibrated_label:
# #                             tmp.append(j)
# #                     S_hat.append(tmp)
# #
# #             # # conformal inference on label sensors
# #             # for i in tqdm(range(num_label)):
# #             #     P_test = y_hat_all[:, i, :]
# #             #     P_test = P_test.detach().cpu()
# #             #
# #             #     rng = np.random.default_rng(args.seed)
# #             #     epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
# #             #     grey_box_test = ProbAccum(P_test)
# #             #     S_hat = grey_box_test.predict_sets(alpha_calibrated_label[i], epsilon=epsilon, allow_empty=False)
# #             #
# #             #     for k, l in enumerate(S_hat):
# #             #         # if size_all[k] == 1:
# #             #         #     continue
# #             #         if len(l) == 1 and l[0] == 0:
# #             #             size_all[k] -= 1
# #             #             y_hat[k].remove(i)
# #             #         # elif len(l) == 1 and l[0] == 1:
# #             #         #     size_all[k] = 1
# #             #         #     y_hat[k] = [i]
# #
# #             # conformal inference on attribute sensors
# #             # print(f'Use knowledge_set_correction: {args.knowledge_set_correction}')
# #             # for i in tqdm(range(num_attribute)):
# #             #     P_test = y_hat_all[:, i + num_label, :]
# #             #     P_test = P_test.detach().cpu()
# #             #
# #             #     rng = np.random.default_rng(args.seed)
# #             #     epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
# #             #     grey_box_test = ProbAccum(P_test)
# #             #     S_hat = grey_box_test.predict_sets(alpha_calibrated_attribute[i], epsilon=epsilon, allow_empty=False)
# #             #
# #             #     if args.knowledge_set_correction:
# #             #         for k, l in enumerate(S_hat):
# #             #             # if size_all[k] == 1:
# #             #             #     continue
# #             #             if len(l) == 1 and l[0] == 0:
# #             #                 for jj in range(len(mappings_attribute[i])):
# #             #                     if mappings_attribute[i][jj] == 1:
# #             #                         if jj in y_hat[k]:
# #             #                             size_all[k] -= 1
# #             #                             y_hat[k].remove(jj)
# #             #             elif len(l) == 1 and l[0] == 1:
# #             #                 for jj in range(len(mappings_attribute[i])):
# #             #                     if mappings_attribute[i][jj] == 0:
# #             #                         if jj in y_hat[k]:
# #             #                             size_all[k] -= 1
# #             #                             y_hat[k].remove(jj)
# #             # evaluation
# #             # avg_size = size_all.sum() / len(size_all)
# #             # marginal_coverage = 0
# #             # for i in range(len(label_test)):
# #             #     if label_test[i] in S_hat[i]:
# #             #         marginal_coverage += 1
# #             # marginal_coverage = marginal_coverage / len(label_test)
# #
# #             # evaluation
# #             total_size = 0
# #             marginal_coverage = 0
# #             for i, l in enumerate(S_hat):
# #                 total_size += len(l)
# #                 if label_test[i] in l:
# #                     marginal_coverage += 1
# #             avg_size = total_size / len(S_hat)
# #             marginal_coverage = marginal_coverage / len(S_hat)
# #             print(f'marginal coverage under attack {args.attack_type}: {marginal_coverage}')
# #             print(f'avg_size under attack {args.attack_type}: {avg_size}')
# #         else:
# #             print(f'args.calibrate={args.calibrate}, args.inference={args.inference} are not satisfactory!')
# #             sys.exit(1)
# #     else:
# #         print(f'Conformal method {args.method_conformal} is not implemented!')
# #         sys.exit(1)