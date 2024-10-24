from dataset.datasets_AwA2 import get_dataset
from torch.utils.data import DataLoader
from model.architectures_AwA2 import get_architecture
import torch
from torch.nn import  Sigmoid, Softmax
from tqdm import tqdm
from dataset.datasets_AwA2 import get_dataset
from torch.utils.data import DataLoader
from model.architectures_AwA2 import get_architecture
import torch
from torch.nn import  Sigmoid, Softmax
from tqdm import tqdm
from torch.utils.data import random_split
from conformal_attack import conformal_attack
from scipy.stats import norm
import numpy as np
from arc.classification import ProbabilityAccumulator as ProbAccum
from dataset.dataset import DataMain
from dataset.dataset import Cifar10
from model.model import NEURAL
from model.model_single import NEURAL_single
from tqdm import tqdm
from scipy.stats.mstats import mquantiles

batch_size = 50
noise_sd = 0.25
num_val = 200
num_test = 200
alpha = 0.1

seed = 2023
num_label = 50
num_attribute = 113

# train_dataset = get_dataset('AWA', 'train')
test_dataset = get_dataset('AWA', 'test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=300,num_workers=8, pin_memory=False)
# val_dataset, test_dataset,_ = random_split(test_dataset, [num_val, num_test,len(test_dataset)-num_val-num_test])
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
#                           num_workers=8, pin_memory=False)
# val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
#                              num_workers=8, pin_memory=False)
#
all_path = [f'/data/common/AwA2/saved_models/noise_sd_{noise_sd:.2f}/main.pth.tar']
all_path.extend([f'/data/common/AwA2/saved_models/noise_sd_{noise_sd:.2f}/attribute_{i}.pth.tar' for i in range(85)])
all_path.extend([f'/data/common/AwA2/saved_models/noise_sd_{noise_sd:.2f}/hierarchy_{i}.pth.tar' for i in range(28)])

matrix = torch.load('/data/common/AwA2/Animals_with_Attributes2/gt_matrix.pt')
label_map = torch.eye(num_label)
attr_map = matrix

test_confidence = torch.load(f'/data/common/AwA2/logs/test_{noise_sd:.2f}.pt')
label_test = test_dataset

# print(test_confidence.shape)
# print(test_confidence[0])

# union bound
alpha = alpha / (num_label + num_attribute)

image_test = []
label_test = []
for x,y in test_loader:
    image_test.append(x)
    label_test.append(y)
label_test = torch.concat(label_test, dim=0)
# print(label_test.shape)
# print(label_test[0])

l = len(label_test)
val_index = np.random.choice(np.array(list(range(0,l//2))),num_val)
test_index = np.random.choice(np.array(list(range(l//2,l))),num_test)

alpha_calibrated_all = []

# calibrate label sensors
for i in tqdm(range(num_label+num_attribute)):
    if i < num_label:
        label_mapping = label_map[:,i]
    else:
        label_mapping = attr_map[:,i-num_label]

    y_hat = test_confidence[val_index, i:i+1]
    y_hat = torch.concat([1.0-y_hat,y_hat],dim=1)
    label = label_mapping[label_test[val_index]]

    y_hat = y_hat.detach().cpu()
    label = label.cpu()

    n2 = y_hat.shape[0]
    grey_box = ProbAccum(y_hat)
    rng = np.random.default_rng(seed)
    epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
    label = label.int()
    alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)

    scores = alpha - alpha_max
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
    alpha_correction = mquantiles(scores, prob=level_adjusted)
    alpha_calibrated_ = alpha - alpha_correction
    alpha_calibrated_all.append(alpha_calibrated_[0])

print(alpha_calibrated_all)


# conformal inference on sensing models
size_all = torch.zeros((len(test_index)))
size_all = size_all + num_label
labels = label_test[test_index]
predict_set = {}
for i in range(len(labels)):
    predict_set[i] = list(range(0, num_label))
for i in tqdm(range(num_label)):
    y_hat = test_confidence[test_index, i:i + 1]
    P_test = torch.concat([1.0 - y_hat, y_hat], dim=1)
    P_test = P_test.detach().cpu()

    rng = np.random.default_rng(seed)
    epsilon = rng.uniform(low=0.0, high=1.0, size=len(P_test))
    grey_box_test = ProbAccum(P_test)
    S_hat = grey_box_test.predict_sets(alpha_calibrated_all[i], epsilon=epsilon, allow_empty=False)

    for k, l in enumerate(S_hat):
        if len(l) == 1 and l[0] == 0:
            size_all[k] -= 1
            predict_set[k].remove(i)
# evaluation
avg_size = size_all.sum() / len(size_all)
marginal_coverage = 0
for i in range(len(labels)):
    if labels[i] in predict_set[i]:
        marginal_coverage += 1
marginal_coverage = marginal_coverage / len(labels)
print(f'marginal coverage: {marginal_coverage}')
print(f'avg_size: {avg_size}')


#
#
#
# logits = []
# labels = []
#
# with torch.no_grad():
#     for i, (inputs, targets) in tqdm(enumerate(val_loader)):
#         inputs, targets = inputs.cuda(), targets.cuda()
#         cur_prediction = []
#         labels.append(targets)
#         for model_id, path in enumerate(all_path):
#             checkpoint = torch.load(path, map_location='cpu')
#             if model_id == 0:
#                 model = get_architecture('resnet50', 'AWA', classes = 50)
#                 m = Softmax(dim=1)
#             else:
#                 model = get_architecture('resnet50', 'AWA', classes = 1)
#                 m = Sigmoid()
#             model.load_state_dict(checkpoint['state_dict'])
#             model = model.cuda()
#             model.eval()
#
#             inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
#             y_hat = m(model(inputs))
#             cur_prediction.append(y_hat)
#         cur_prediction = torch.concat(cur_prediction,dim=1)
#         logits.append(cur_prediction)
#
# logits = torch.concat(logits, dim=0)
#
#
# # calibrate label sensors
# for i in tqdm(range(num_label)):
#     label_mapping = mappings_label[i]
#     cur = 0
#     y_hat = y_hat_all[:, i]
#     label = torch.zeros((len(data.y_val))).cuda()
#     X, GT = data.sequential_val_batch()
#     while X is not None:
#         X = X.cuda()
#         GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
#         this_batch_size = len(X)
#         label[cur:cur + this_batch_size] = GT
#         cur = cur + this_batch_size
#         X, GT = data.sequential_val_batch()
#     y_hat = y_hat.detach().cpu()
#     label = label.cpu()
#
#     n2 = y_hat.shape[0]
#     grey_box = ProbAccum(y_hat)
#     rng = np.random.default_rng(args.seed)
#     epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
#     label = label.int()
#     alpha_max = grey_box.calibrate_scores(label, epsilon=epsilon)
#
#     scores = args.alpha - alpha_max
#     level_adjusted = (1.0 - args.alpha) * (1.0 + 1.0 / float(n2))
#     alpha_correction = mquantiles(scores, prob=level_adjusted)
#     alpha_calibrated_ = args.alpha - alpha_correction
#     alpha_calibrated_label.append(alpha_calibrated_[0])
