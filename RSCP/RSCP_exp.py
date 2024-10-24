# general imports
import gc
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import random
import torch
import torchvision
import os
import pickle
import sys
import argparse
from torchvision import transforms, datasets
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import pandas as pd

# My imports
sys.path.insert(0, './')
from Third_Party.smoothing_adversarial.architectures import get_architecture
import RSCP.Score_Functions as scores
from RSCP.utils import evaluate_predictions,calculate_accuracy_smooth, \
    smooth_calibration_ImageNet, predict_sets_ImageNet, Smooth_Adv_ImageNet, get_scores, get_normalize_layer, calibration, prediction
from Architectures.DenseNet import DenseNet
from Architectures.VGG import vgg19_bn, VGG
from Architectures.ResNet import ResNet

# parameters
parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-d', '--delta', default=0.125, type=float, help='L2 bound on the adversarial noise')
parser.add_argument('-s', '--splits', default=50, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('-r', '--ratio', default=2, type=float,
                    help='Ratio between adversarial noise bound to smoothing noise')
parser.add_argument('--n_s', default=256, type=int, help='Number of samples used for estimating smoothed score')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to be used: CIFAR100, CIFAR10, ImageNet')
parser.add_argument('--arc', default='ResNet', type=str,
                    help='Architecture of classifier : ResNet, DenseNet, VGG. Relevant only of My_model=True')
parser.add_argument('--My_model', action='store_true', help='True for our trained model, False for Cohens. Relevent only for CIFAR10')
parser.add_argument('--batch_size', default=1024, type=int, help='Number of images to send to gpu at once')
parser.add_argument('--sigma_model', default=-1, type=float, help='std of Gaussian noise the model was trained with')
parser.add_argument('--Salman', action='store_true', help='True for Salman adversarial model, False for Cohens. Relevent only for CIFAR10')
parser.add_argument('--coverage_on_label', action='store_true', help='True for getting coverage and size for each label')
args = parser.parse_args()

# parameters
alpha = args.alpha  # desired nominal marginal coverage
epsilon = args.delta  # L2 bound on the adversarial noise
n_experiments = args.splits  # number of experiments to estimate coverage
ratio = args.ratio  # ratio between adversarial noise bound to smoothed noise
sigma_smooth = ratio * epsilon # sigma used fro smoothing
# sigma used for training the model
if args.sigma_model != -1:
    sigma_model = args.sigma_model
else:
    sigma_model = sigma_smooth
n_smooth = args.n_s  # number of samples used for smoothing
My_model = args.My_model
N_steps = 20  # number of gradiant steps for PGD attack
dataset = args.dataset  # dataset to be used  CIFAR100', 'CIFAR10', 'ImageNet'
calibration_scores = ['HCC', 'SC']  # score function to check 'HCC', 'SC', 'SC_Reg'
model_type = args.arc # Architecture of the model
Salman = args.Salman # Whether to use Salman adversarial model or not
coverage_on_label = args.coverage_on_label # Whether to calculate coverage and size per class

# number of test points (if larger then available it takes the entire set)
if dataset == 'ImageNet':
    n_test = 50000
else:
    n_test = 10000

# Validate parameters
assert dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'ImageNet', 'Dataset can only be CIFAR10 or CIFAR100 or ImageNet.'
assert 0 <= alpha <= 1, 'Nominal level must be between 0 to 1'
assert not(n_smooth & (n_smooth - 1)), 'n_s must be a power of 2.'
assert not(args.batch_size & (args.batch_size - 1)), 'batch size must be a power of 2.'
assert args.batch_size >= n_smooth, 'batch size must be larger than n_s'
assert model_type == 'ResNet' or model_type == 'DenseNet' or model_type == 'VGG', 'Architecture can only be Resnet, ' \
                                                                                  'VGG or DenseNet '
assert sigma_model >= 0, 'std for training the model must be a non negative number.'
assert epsilon >= 0, 'L2 bound of noise must be non negative.'
assert isinstance(n_experiments, int) and n_experiments >= 1, 'number of splits must be a positive integer.'
assert ratio >= 0, 'ratio between sigma and delta must be non negative.'

# CIFAR100 has only my models
if dataset == "CIFAR100":
    My_model = True

# All our models are needs to be added a normalization layer
if My_model:
    normalized = True

# Cohen models already have this layer, Plus only ResNet is available for them.
else:
    normalized = False
    model_type = 'ResNet'

# The GPU used for oue experiments can only handle the following quantities of images per batch
GPU_CAPACITY = args.batch_size

# Save results to final results directories only if full data is taken. Otherwise save locally.
if ((dataset == 'ImageNet') and (n_experiments == 50) and (n_test == 50000))\
        or ((dataset != 'ImageNet') and (n_experiments == 50) and (n_test == 10000)):
    save_results = True
else:
    save_results = False

# calculate correction based on the Lipschitz constant
if sigma_smooth == 0:
    correction = 10000
else:
    correction = float(epsilon) / float(sigma_smooth)

# set random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load datasets
if dataset == "CIFAR10":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR10(root='./datasets/',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor())

elif dataset == "CIFAR100":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR100(root='./datasets/',
                                                  train=True,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                                 train=False,
                                                 transform=torchvision.transforms.ToTensor())
elif dataset == "ImageNet":
    # get dir of imagenet validation set
    imagenet_dir = "./datasets/imagenet"

    # ImageNet images pre-processing
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    # load dataset
    test_dataset = datasets.ImageFolder(imagenet_dir, transform)

else:
    print("No such dataset")
    exit(1)

# cut the size of the test set if necessary
if n_test < len(test_dataset):
    torch.manual_seed(0)
    test_dataset = torch.utils.data.random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]

# save the sizes of each one of the sets
n_test = len(test_dataset)

# Create Data loader for test set
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=n_test,
                                          shuffle=False)

# convert test set into tensor
examples = enumerate(test_loader)
batch_idx, (x_test, y_test) = next(examples)

# get dimension of data
rows = x_test.size()[2]
cols = x_test.size()[3]
channels = x_test.size()[1]
if dataset == 'ImageNet':
    num_of_classes = 1000
elif dataset == 'CIFAR100':
    num_of_classes = 100
else:
    num_of_classes = 10
min_pixel_value = 0.0
max_pixel_value = 1.0

# automatically choose device use gpu 0 if it is available o.w. use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print the chosen device
print("device: ", device)

# load my models
if My_model:
    if dataset == "CIFAR10":
        if model_type == 'ResNet':
            model = ResNet(depth=110, num_classes=10)
            state = torch.load('./checkpoints/CIFAR10_ResNet110_Robust_sigma_' + str(sigma_model) + '.pth.tar',
                               map_location=device)
        elif model_type == 'DenseNet':
            model = DenseNet(depth=100, num_classes=10, growthRate=12)
            state = torch.load('./checkpoints/CIFAR10_DenseNet_sigma_' + str(sigma_model) + '.pth.tar',
                               map_location=device)
        elif model_type == 'VGG':
            model = vgg19_bn(num_classes=10)
            state = torch.load('./checkpoints/CIFAR10_VGG_sigma_' + str(sigma_model) + '.pth.tar', map_location=device)
        else:
            print("No such architecture")
            exit(1)
        normalize_layer = get_normalize_layer("cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        model.load_state_dict(state['state_dict'])
    elif dataset == "CIFAR100":
        if model_type == 'ResNet':
            model = ResNet(depth=110, num_classes=100)
            state = torch.load('./checkpoints/ResNet110_Robust_sigma_' + str(sigma_model) + '.pth.tar',
                               map_location=device)
        elif model_type == 'DenseNet':
            model = DenseNet(depth=100, num_classes=100, growthRate=12)
            state = torch.load('./checkpoints/DenseNet_sigma_' + str(sigma_model) + '.pth.tar', map_location=device)
        elif model_type == 'VGG':
            model = vgg19_bn(num_classes=100)
            state = torch.load('./checkpoints/VGG_sigma_' + str(sigma_model) + '.pth.tar', map_location=device)
        else:
            print("No such architecture")
            exit(1)
        normalize_layer = get_normalize_layer("cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        model.load_state_dict(state['state_dict'])
    else:
        print("No My model exist for ImageNet dataset")
        exit(1)

# load cohen and salman models
else:
    if dataset == "CIFAR10":
        if Salman:
            #checkpoint = torch.load(
            #    './Pretrained_Models/Salman/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_'+str(sigma_model)+'/checkpoint.pth.tar', map_location=device)
            checkpoint = torch.load('./Pretrained_Models/Salman/cifar10/PGD_10steps_multiNoiseSamples/2-multitrain/eps_32/cifar10/resnet110/noise_'+str(sigma_model)+'/checkpoint.pth.tar', map_location=device)
        else:
            checkpoint = torch.load(
                './Pretrained_Models/Cohen/cifar10/resnet110/noise_' + str(sigma_model) + '/checkpoint.pth.tar', map_location=device)
        model = get_architecture(checkpoint["arch"], "cifar10")
    elif dataset == "ImageNet":
        checkpoint = torch.load(
            './Pretrained_Models/Cohen/imagenet/resnet50/noise_' + str(sigma_model) + '/checkpoint.pth.tar',
            map_location=device)
        model = get_architecture(checkpoint["arch"], "imagenet")
    else:
        print("No Cohens model for CIFAR100")
        exit(1)
    model.load_state_dict(checkpoint['state_dict'])


# send model to device
model.to(device)

# put model in evaluation mode
model.eval()

# create indices for the test points
indices = torch.arange(n_test)

# directory to store adversarial examples and noises
directory = "./Adversarial_Examples/" + str(dataset) + "/epsilon_" + str(epsilon) + "/sigma_model_" + str(
        sigma_model) + "/sigma_smooth_" + str(sigma_smooth) + "/n_smooth_" + str(n_smooth)

# normalization layer to my model
if normalized:
    directory = directory + "/Robust"

# different attacks for different architectures
if model_type != 'ResNet':
    directory = directory + "/" + str(model_type)

# different attacks for my or cohens and salman model
if dataset == "CIFAR10" and model_type == 'ResNet':
    if My_model:
        directory = directory + "/My_Model"
    else:
        directory = directory + "/Their_Model"
        if Salman:
            directory = directory + "/Salman"

print("Save results on directories: " + str(save_results))
print("Searching for adversarial examples in: " + str(directory))
if os.path.exists(directory):
    print("Are there saved adversarial examples: Yes")
else:
    print("Are there saved adversarial examples: No")

# If there are no pre created adversarial examples, create new ones
if ((dataset != 'ImageNet') and (n_test != 10000)) or ((dataset == 'ImageNet') and (n_test != 50000)) or not os.path.exists(directory):
    # Generate adversarial test examples
    print("Generate adversarial test examples for the smoothed model:\n")
    x_test_adv = Smooth_Adv_ImageNet(model, x_test, y_test, indices, n_smooth, sigma_smooth, N_steps, epsilon, device, GPU_CAPACITY=GPU_CAPACITY)

    # Generate adversarial test examples for the base classifier
    print("Generate adversarial test examples for the base model:\n")
    x_test_adv_base = Smooth_Adv_ImageNet(model, x_test, y_test, indices, 1, sigma_model, N_steps, epsilon, device,
                                GPU_CAPACITY=GPU_CAPACITY)

    # Only store examples for full dataset
    if ((dataset == 'ImageNet') and (n_test == 50000)) \
            or ((dataset != 'ImageNet') and (n_test == 10000)):
        os.makedirs(directory)
        with open(directory + "/data.pickle", 'wb') as f:
            pickle.dump([x_test_adv, x_test_adv_base], f)

# If there are pre created adversarial examples, load them
else:
    with open(directory + "/data.pickle", 'rb') as f:
        x_test_adv, x_test_adv_base = pickle.load(f)


# create the noises for the base classifiers only to check its accuracy
noises_base = torch.empty_like(x_test)
for k in range(n_test):
    torch.manual_seed(k)
    noises_base[k:(k + 1)] = torch.randn(
        (1, channels, rows, cols)) * sigma_model

# Calculate accuracy of classifier on clean test points
acc, _, _ = calculate_accuracy_smooth(model, x_test, y_test, noises_base, num_of_classes, k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy :" + str(acc * 100) + "%")

# Calculate accuracy of classifier on adversarial test points
acc, _, _ = calculate_accuracy_smooth(model, x_test_adv_base, y_test, noises_base, num_of_classes, k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy on adversarial examples :" + str(acc * 100) + "%")
#exit(1)

del noises_base
gc.collect()

# translate desired scores to their functions and put in a list
scores_list = []
for score in calibration_scores:
    if score == 'HCC':
        scores_list.append(scores.class_probability_score)
    elif score == 'SC':
        scores_list.append(scores.generalized_inverse_quantile_score)
    elif score == 'SC_Reg':
        scores_list.append(scores.rank_regularized_score)
    else:
        print("Undefined score function")
        exit(1)

print("Calculating scores for entire dataset:\n")

# get base scores of whole clean test set
print("Calculating base scores on the clean test points:\n")
scores_simple_clean_test = get_scores(model, x_test, indices, n_smooth, sigma_model, num_of_classes, scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)

# get smooth scores of whole clean test set
print("Calculating smoothed scores on the clean test points:\n")
smoothed_scores_clean_test, scores_smoothed_clean_test = get_scores(model, x_test, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)

# get base scores of whole clean adversarial set
print("Calculating base scores on the adversarial test points:\n")
scores_simple_adv_test = get_scores(model, x_test_adv_base, indices, n_smooth, sigma_model, num_of_classes, scores_list, base=True, device=device, GPU_CAPACITY=GPU_CAPACITY)

# get smooth scores of whole adversarial test set
print("Calculating smoothed scores on the adversarial test points:\n")
smoothed_scores_adv_test, scores_smoothed_adv_test = get_scores(model, x_test_adv, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, base=False, device=device, GPU_CAPACITY=GPU_CAPACITY)

# save scores
# one_hot_lables = np.zeros((n_test, num_of_classes))
# flattend_input = np.zeros((n_test, (x_test[0].flatten()).size()[0]))
# flattend_input_adv = np.zeros((n_test, (x_test_adv[0].flatten()).size()[0]))
# for i in range(n_test):
#     one_hot_lables[i, y_test[i]] = 1
#     flattend_input[i, :] = np.array(x_test[i].flatten())
#     flattend_input_adv[i, :] = np.array(x_test_adv[i].flatten())
# dict_to_save_clean = {'labels': one_hot_lables, 'scores_HPS': scores_simple_clean_test[0],'scores_APS': scores_simple_clean_test[1],'scores_smoothed_HPS': np.zeros((0,0)),'scores_smoothed_APS': np.zeros((0,0)), 'features': flattend_input}
# dict_to_save_adv = {'labels': one_hot_lables, 'scores_HPS': scores_simple_adv_test[0],'scores_APS': scores_simple_adv_test[1],'scores_smoothed_HPS': np.zeros((0,0)),'scores_smoothed_APS': np.zeros((0,0)), 'features': flattend_input_adv}
#
#
# np.save("regular_model_clean_test", dict_to_save_clean)
# np.save("regular_model_adv_test", dict_to_save_adv)
# exit(1)

# clean unnecessary data
del x_test, x_test_adv, x_test_adv_base
gc.collect()

# create dataframe for storing results
results = pd.DataFrame()

# container for storing bounds on "CP+SS"
quantiles = np.zeros((len(scores_list), 2, n_experiments))

# run for n_experiments data splittings
print("\nRunning experiments for "+str(n_experiments)+" random splits:\n")
for experiment in tqdm(range(n_experiments)):

    # Split test data into calibration and test
    idx1, idx2 = train_test_split(indices, test_size=0.5)

    # calibrate base model with the desired scores and get the thresholds
    thresholds_base, _ = calibration(scores_simple=scores_simple_clean_test[:, idx1, y_test[idx1]], alpha=alpha, num_of_scores=len(scores_list), correction=correction, base=True)

    # calibrate the model with the desired scores and get the thresholds
    thresholds, bounds = calibration(scores_smoothed=scores_smoothed_clean_test[:, idx1, y_test[idx1]], smoothed_scores=smoothed_scores_clean_test[:, idx1, y_test[idx1]], alpha=alpha, num_of_scores=len(scores_list), correction=correction, base=False)

    thresholds = thresholds + thresholds_base

    # put bounds in array of bounds
    for p in range(len(scores_list)):
        quantiles[p, 0, experiment] = bounds[p, 0]
        quantiles[p, 1, experiment] = bounds[p, 1]

    # generate prediction sets on the clean test set for base model
    predicted_clean_sets_base = prediction(scores_simple=scores_simple_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds, base=True)

    # generate robust prediction sets on the clean test set
    predicted_clean_sets = prediction(scores_smoothed=scores_smoothed_clean_test[:, idx2, :], smoothed_scores=smoothed_scores_clean_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds, correction=correction, base=False)

    # generate prediction sets on the adversarial test set for base model
    predicted_adv_sets_base = prediction(scores_simple=scores_simple_adv_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds, base=True)

    # generate robust prediction sets on the adversarial test set
    predicted_adv_sets = prediction(scores_smoothed=scores_smoothed_adv_test[:, idx2, :], smoothed_scores=smoothed_scores_adv_test[:, idx2, :], num_of_scores=len(scores_list), thresholds=thresholds, correction=correction, base=False)

    # arrange results on clean test set in dataframe
    for p in range(len(scores_list)):
        predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])
        predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
        score_name = calibration_scores[p]
        methods_list = [score_name + '_simple', score_name + '_smoothed_classifier', score_name + '_smoothed_score',
                        score_name + '_smoothed_score_correction']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_clean_sets[p][r], None, y_test[idx2].numpy(),
                                       conditional=False,coverage_on_label=coverage_on_label, num_of_classes=num_of_classes)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = 0
            res['Black box'] = 'CNN sigma = ' + str(sigma_model)
            # Add results to the list
            results = results.append(res)

    # arrange results on adversarial test set in dataframe
    for p in range(len(scores_list)):
        score_name = calibration_scores[p]
        methods_list = [score_name + '_simple', score_name + '_smoothed_classifier', score_name + '_smoothed_score',
                        score_name + '_smoothed_score_correction']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_adv_sets[p][r], None, y_test[idx2].numpy(),
                                       conditional=False, coverage_on_label=coverage_on_label, num_of_classes=num_of_classes)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = epsilon
            res['Black box'] = 'CNN sigma = ' + str(sigma_model)
            # Add results to the list
            results = results.append(res)

    # clean memory
    del idx1, idx2, predicted_clean_sets, predicted_clean_sets_base, predicted_adv_sets, predicted_adv_sets_base, bounds, thresholds, thresholds_base
    gc.collect()

# add given y string at the end
if coverage_on_label:
    add_string = "_given_y"
else:
    add_string = ""

if save_results:
    # directory to save results
    directory = "./Results/" + str(dataset) + "/epsilon_" + str(epsilon) + "/sigma_model_" + str(
        sigma_model) + "/sigma_smooth_" + str(sigma_smooth) + "/n_smooth_" + str(n_smooth)

    if normalized:
        directory = directory + "/Robust"

    if model_type != 'ResNet':
        directory = directory + "/" + str(model_type)

    if dataset == "CIFAR10" and model_type == 'ResNet':
        if My_model:
            directory = directory + "/My_Model"
        else:
            directory = directory + "/Their_Model"
            if Salman:
                directory = directory + "/Salman"

    for score in calibration_scores:
        if score == 'SC_Reg':
            directory = directory + "/Regularization"
            break

    if alpha != 0.1:
        directory = directory + "/alpha_" + str(alpha)

    print("Saving results in: " + str(directory))

    if not os.path.exists(directory):
        os.makedirs(directory)

    # save results
    results.to_csv(directory + "/results"+add_string+".csv")
    with open(directory + "/quantiles_bounds"+add_string+".pickle", 'wb') as f:
        pickle.dump([quantiles], f)
else:
    # save results
    print("Saving results in main Results folder")
    results.to_csv("./Results/results"+add_string+".csv")

# plot results
# plot marginal coverage results
colors_list = sns.color_palette("husl", len(scores_list) * 4)

ax = sns.catplot(x="Black box", y="Coverage",
                 hue="Method", palette=colors_list, col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)

lower_quantiles_mean = np.zeros(len(scores_list))
upper_quantiles_mean = np.zeros(len(scores_list))
lower_quantiles_std = np.zeros(len(scores_list))
upper_quantiles_std = np.zeros(len(scores_list))

for p in range(len(scores_list)):
    lower_quantiles_mean[p] = np.mean(quantiles[p, 0, :])
    upper_quantiles_mean[p] = np.mean(quantiles[p, 1, :])
    lower_quantiles_std[p] = np.std(quantiles[p, 0, :])
    upper_quantiles_std[p] = np.std(quantiles[p, 1, :])

colors = ['green', 'blue']
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Marginal coverage')
    graph.axhline(1 - alpha, ls='--', color="red")
    for p in range(len(scores_list)):
        graph.axhline(upper_quantiles_mean[p], ls='--', color=colors_list[p * 4 + 2])
        graph.axhline(lower_quantiles_mean[p], ls='--', color=colors_list[p * 4 + 2])

if save_results:
    ax.savefig(directory + "/Marginal"+add_string+".pdf")
else:
    ax.savefig("./Results/Marginal"+add_string+".pdf")

# plot set sizes results
ax = sns.catplot(x="Black box", y="Size",
                 hue="Method", col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Set Size')

if save_results:
    ax.savefig(directory + "/Size"+add_string+".pdf")
else:
    ax.savefig("./Results/Size"+add_string+".pdf")