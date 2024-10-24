import torch
import numpy as np
import gc
import pandas as pd
from torch.nn.functional import softmax
from scipy.stats import rankdata
from numpy.random import default_rng
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from tqdm import tqdm
from typing import List


# function to calculate accuracy of the model
def calculate_accuracy(model, dataloader, device):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    model_accuracy = total_correct / total_images
    return model_accuracy


def Smooth_Adv(model, x, y, noises, N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024, method='PGD'):
    # create attack model
    if method == 'PGD':
        attacker = PGD_L2(steps=N_steps, device=device, max_norm=max_norm)
    elif method == "DDN":
        attacker = DDN(steps=N_steps, device=device, max_norm=max_norm)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # number of permutations to estimate mean
    num_of_noise_vecs = noises.size()[0] // n

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // num_of_noise_vecs

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    print("Generating Adverserial Examples:")

    for j in tqdm(range(num_of_batches)):
        #GPUtil.showUtilization()
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first num_of_noise_vecs samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * num_of_noise_vecs, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, num_of_noise_vecs, 1, 1)).view(tmp.shape).to(device)

        # send labels to device
        y_tmp = labels.to(device).long()

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * num_of_noise_vecs)):((j + 1) * (batch_size * num_of_noise_vecs))].to(device)
        # noise = torch.randn_like(x_tmp, device=device) * sigma_adv

        # generate adversarial examples for the batch
        x_adv_batch = attacker.attack(model, x_tmp, y_tmp,
                                      noise=noise, num_noise_vectors=num_of_noise_vecs,
                                      no_grad=False,
                                      )

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::num_of_noise_vecs]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()


    # return adversarial examples
    return x_adv


def evaluate_predictions(S, X, y, conditional=False, coverage_on_label=False, num_of_classes=10):

    # get numbers of points
    #n = np.shape(X)[0]

    # get points to a matrix of the format nxp
    #X = np.vstack([X[i, 0, :, :].flatten() for i in range(n)])

    # Marginal coverage
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])

    # If desired calculate coverage for each class
    if coverage_on_label:
        sums = np.zeros(num_of_classes)
        size_sums = np.zeros(num_of_classes)
        lengths = np.zeros(num_of_classes)
        for i in range(len(y)):
            lengths[y[i]] = lengths[y[i]] + 1
            size_sums[y[i]] = size_sums[y[i]] + len(S[i])
            if y[i] in S[i]:
                sums[y[i]] = sums[y[i]] + 1
        coverage_given_y = sums/lengths
        lengths_given_y = size_sums/lengths

    # Conditional coverage not implemented
    wsc_coverage = None

    # Size and size conditional on coverage
    size = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    size_cover = np.mean([len(S[i]) for i in idx_cover])

    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Size': [size], 'Size cover': [size_cover]})

    # If desired, save coverage for each class
    if coverage_on_label:
        for i in range(num_of_classes):
            out['Coverage given '+str(i)] = coverage_given_y[i]
            out['Size given '+str(i)] = lengths_given_y[i]

    return out


# calculate accuracy of the smoothed classifier
def calculate_accuracy_smooth(model, x, y, noises, num_classes, k=1, device='cpu', GPU_CAPACITY=1024):
    # get size of the test set
    n = x.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n

    # create container for the outputs
    smoothed_predictions = torch.zeros((n, num_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # get predictions over all batches
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first n_smooth samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

        # add noise to points
        noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1)

        # get smoothed prediction for each point
        for m in range(len(labels)):
            smoothed_predictions[(j * batch_size) + m, :] = torch.mean(
                noisy_outputs[(m * n_smooth):((m + 1) * n_smooth)], dim=0)

    # transform results to numpy array
    smoothed_predictions = smoothed_predictions.numpy()

    # get label ranks to calculate top k accuracy
    label_ranks = np.array([rankdata(-smoothed_predictions[i, :], method='ordinal')[y[i]] - 1 for i in range(n)])

    # get probabilities of correct labels
    label_probs = np.array([smoothed_predictions[i, y[i]] for i in range(n)])

    # calculate accuracy
    top_k_accuracy = np.sum(label_ranks <= (k - 1)) / float(n)

    # calculate average inverse probability score
    score = np.mean(1 - label_probs)

    # calculate the 90 qunatiule
    quantile = mquantiles(1-label_probs, prob=0.9)
    return top_k_accuracy, score, quantile


def smooth_calibration(model, x_calib, y_calib, noises, alpha, num_of_classes, scores_list, correction, base=False, device='cpu', GPU_CAPACITY=1024):
    # size of the calibration set
    n_calib = x_calib.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n_calib

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n_calib))
    else:
        smoothed_scores = np.zeros((len(scores_list), n_calib))
        scores_smoothed = np.zeros((len(scores_list), n_calib))

    # create container for the calibration thresholds
    thresholds = np.zeros((len(scores_list), 3))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n_calib % batch_size != 0:
        num_of_batches = (n_calib // batch_size) + 1
    else:
        num_of_batches = (n_calib // batch_size)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n_calib, num_of_classes))
    else:
        smooth_outputs = np.zeros((n_calib, num_of_classes))

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n_calib, low=0.0, high=1.0)

    # pass all points to model in batches and calculate scores
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x_calib[(j * batch_size):((j + 1) * batch_size)]
        labels = y_calib[(j * batch_size):((j + 1) * batch_size)]

        if base:
            noise = noises[(j * batch_size):((j + 1) * batch_size)].to(device)
            noisy_points = inputs.to(device) + noise
        else:
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # generate random Gaussian noise for the duplicated batch
            noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

            # add noise to points
            noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        # get smoothed score for each point
        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            for k in range(len(labels)):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # get smoothed score of this point

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores
                for p, score_func in enumerate(scores_list):
                    # get smoothed score
                    tmp_scores = score_func(point_outputs, labels[k], u, all_combinations=True)
                    smoothed_scores[p, (j * batch_size) + k] = np.mean(tmp_scores)

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :] = score_func(simple_outputs, y_calib, uniform_variables, all_combinations=False)
        else:
            scores_smoothed[p, :] = score_func(smooth_outputs, y_calib, uniform_variables, all_combinations=False)

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    bounds = np.zeros((len(scores_list), 2))
    for p in range(len(scores_list)):
        if base:
            thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        else:
            thresholds[p, 1] = mquantiles(scores_smoothed[p, :], prob=level_adjusted)
            thresholds[p, 2] = mquantiles(smoothed_scores[p, :], prob=level_adjusted)

            # calculate lower and upper bounds of correction of smoothed score
            upper_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)+correction, loc=0, scale=1)
            lower_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)-correction, loc=0, scale=1)

            bounds[p, 0] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= lower_thresh])/np.size(smoothed_scores[p, :])
            bounds[p, 1] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= upper_thresh]) / np.size(smoothed_scores[p, :])

    return thresholds, bounds


def smooth_calibration_ImageNet(model, x_calib, y_calib, n_smooth, sigma_smooth, alpha, num_of_classes, scores_list, correction, base=False, device='cpu', GPU_CAPACITY=1024):
    # size of the calibration set
    n_calib = x_calib.size()[0]

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n_calib))
    else:
        smoothed_scores = np.zeros((len(scores_list), n_calib))
        scores_smoothed = np.zeros((len(scores_list), n_calib))

    # create container for the calibration thresholds
    thresholds = np.zeros((len(scores_list), 3))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n_calib % batch_size != 0:
        num_of_batches = (n_calib // batch_size) + 1
    else:
        num_of_batches = (n_calib // batch_size)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n_calib, num_of_classes))
    else:
        smooth_outputs = np.zeros((n_calib, num_of_classes))

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n_calib, low=0.0, high=1.0)

    # pass all points to model in batches and calculate scores
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x_calib[(j * batch_size):((j + 1) * batch_size)]
        labels = y_calib[(j * batch_size):((j + 1) * batch_size)]

        if base:
            noise = (torch.randn_like(inputs)*sigma_smooth).to(device)
            noisy_points = inputs.to(device) + noise
        else:
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # generate random Gaussian noise for the duplicated batch
            noise = (torch.randn_like(x_tmp)*sigma_smooth).to(device)

            # add noise to points
            noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        # get smoothed score for each point
        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            for k in range(len(labels)):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # get smoothed score of this point

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores
                for p, score_func in enumerate(scores_list):
                    # get smoothed score
                    tmp_scores = score_func(point_outputs, labels[k], u, all_combinations=True)
                    smoothed_scores[p, (j * batch_size) + k] = np.mean(tmp_scores)

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :] = score_func(simple_outputs, y_calib, uniform_variables, all_combinations=False)
        else:
            scores_smoothed[p, :] = score_func(smooth_outputs, y_calib, uniform_variables, all_combinations=False)

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    bounds = np.zeros((len(scores_list), 2))
    for p in range(len(scores_list)):
        if base:
            thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        else:
            thresholds[p, 1] = mquantiles(scores_smoothed[p, :], prob=level_adjusted)
            thresholds[p, 2] = mquantiles(smoothed_scores[p, :], prob=level_adjusted)

            # calculate lower and upper bounds of correction of smoothed score
            upper_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)+correction, loc=0, scale=1)
            lower_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)-correction, loc=0, scale=1)

            bounds[p, 0] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= lower_thresh])/np.size(smoothed_scores[p, :])
            bounds[p, 1] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= upper_thresh]) / np.size(smoothed_scores[p, :])

    return thresholds, bounds


def predict_sets(model, x, noises, num_of_classes, scores_list, thresholds, correction, base=False, device='cpu', GPU_CAPACITY=1024):
    # get number of points
    n = x.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        scores_smoothed = np.zeros((len(scores_list), n, num_of_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n, num_of_classes))
    else:
        smooth_outputs = np.zeros((n, num_of_classes))

    for j in range(num_of_batches):
        # get inputs of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]

        if base:
            noise = noises[(j * batch_size):((j + 1) * batch_size)].to(device)
            noisy_points = inputs.to(device) + noise
        else:
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # generate random Gaussian noise for the duplicated batch
            noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

            # add noise to points
            noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            # get smoothed score for each point
            for k in range(inputs.size()[0]):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores with all lables
                for p, score_func in enumerate(scores_list):
                    smoothed_scores[p, ((j * batch_size) + k), :] = np.mean(
                        score_func(point_outputs, np.arange(num_of_classes), u, all_combinations=True), axis=0)

                #return smoothed_scores

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
        else:
            scores_smoothed[p, :, :] = score_func(smooth_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)

    #return scores_simple

    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(len(scores_list)):
        if base:
            S_hat_simple = [np.where(norm.ppf(scores_simple[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 0], loc=0, scale=1))[0] for i in range(n)]
            predicted_sets.append(S_hat_simple)
        else:
            S_hat_smoothed = [np.where(norm.ppf(scores_smoothed[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat_corrected = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) - correction <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]

            tmp_list = [S_hat_smoothed, smoothed_S_hat, smoothed_S_hat_corrected]
            predicted_sets.append(tmp_list)

    # return predictions sets
    return predicted_sets


def predict_sets_ImageNet(model, x, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, thresholds, correction, base=False, device='cpu', GPU_CAPACITY=1024):
    # get number of points
    n = x.size()[0]

    # get dimension of data
    rows = x.size()[2]
    cols = x.size()[3]
    channels = x.size()[1]

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        scores_smoothed = np.zeros((len(scores_list), n, num_of_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n, num_of_classes))
    else:
        smooth_outputs = np.zeros((n, num_of_classes))

    image_index = -1
    for j in range(num_of_batches):

        # get inputs of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        curr_batch_size = inputs.size()[0]

        if base:
            noises_test_base = torch.empty((curr_batch_size, channels, rows, cols))
            # get relevant noises for this batch
            for k in range(curr_batch_size):
                image_index = image_index + 1
                torch.manual_seed(indices[image_index])
                noises_test_base[k:(k + 1)] = torch.randn((1, channels, rows, cols)) * sigma_smooth

            noisy_points = inputs.to(device) + noises_test_base.to(device)
        else:
            noises_test = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
            # get relevant noises for this batch
            for k in range(curr_batch_size):
                image_index = image_index + 1
                torch.manual_seed(indices[image_index])
                noises_test[(k * n_smooth):(k + 1) * n_smooth] = torch.randn(
                    (n_smooth, channels, rows, cols)) * sigma_smooth

            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # add noise to points
            noisy_points = x_tmp + noises_test.to(device)

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            # get smoothed score for each point
            for k in range(inputs.size()[0]):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores with all lables
                for p, score_func in enumerate(scores_list):
                    smoothed_scores[p, ((j * batch_size) + k), :] = np.mean(
                        score_func(point_outputs, np.arange(num_of_classes), u, all_combinations=True), axis=0)
                del u
                gc.collect()

        if base:
            del noisy_points, noisy_outputs, noises_test_base
        else:
            del noisy_points, noisy_outputs, noises_test, tmp
        gc.collect()

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
        else:
            scores_smoothed[p, :, :] = score_func(smooth_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)

    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(len(scores_list)):
        if base:
            S_hat_simple = [np.where(norm.ppf(scores_simple[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 0], loc=0, scale=1))[0] for i in range(n)]
            predicted_sets.append(S_hat_simple)
        else:
            S_hat_smoothed = [np.where(norm.ppf(scores_smoothed[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat_corrected = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) - correction <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]

            tmp_list = [S_hat_smoothed, smoothed_S_hat, smoothed_S_hat_corrected]
            predicted_sets.append(tmp_list)

    # return predictions sets
    return predicted_sets


def Smooth_Adv_ImageNet(model, x, y, indices, n_smooth, sigma_smooth, N_steps=20, max_norm=0.125, device='cpu', GPU_CAPACITY=1024, method='PGD'):
    # create attack model
    if method == 'PGD':
        attacker = PGD_L2(steps=N_steps, device=device, max_norm=max_norm)
    elif method == "DDN":
        attacker = DDN(steps=N_steps, device=device, max_norm=max_norm)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # get dimension of data
    rows = x.size()[2]
    cols = x.size()[3]
    channels = x.size()[1]

    # number of permutations to estimate mean
    num_of_noise_vecs = n_smooth

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // num_of_noise_vecs

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    print("Generating Adverserial Examples:")

    image_index = -1
    for j in tqdm(range(num_of_batches)):
        #GPUtil.showUtilization()
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]
        curr_batch_size = inputs.size()[0]

        # duplicate batch according to the number of added noises and send to device
        # the first num_of_noise_vecs samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * num_of_noise_vecs, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, num_of_noise_vecs, 1, 1)).view(tmp.shape).to(device)

        # send labels to device
        y_tmp = labels.to(device).long()

        # generate random Gaussian noise for the duplicated batch
        noise = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
        # get relevant noises for this batch
        for k in range(curr_batch_size):
            image_index = image_index + 1
            torch.manual_seed(indices[image_index])
            noise[(k * n_smooth):((k + 1) * n_smooth)] = torch.randn(
                (n_smooth, channels, rows, cols)) * sigma_smooth


        #noise = noises[(j * (batch_size * num_of_noise_vecs)):((j + 1) * (batch_size * num_of_noise_vecs))].to(device)
        # noise = torch.randn_like(x_tmp, device=device) * sigma_adv

        noise = noise.to(device)
        # generate adversarial examples for the batch
        x_adv_batch = attacker.attack(model, x_tmp, y_tmp,
                                      noise=noise, num_noise_vectors=num_of_noise_vecs,
                                      no_grad=False,
                                      )

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::num_of_noise_vecs]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()

        del noise, tmp, x_adv_batch
        gc.collect()

    # return adversarial examples
    return x_adv


_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


def get_scores(model, x, indices, n_smooth, sigma_smooth, num_of_classes, scores_list, base=False, device='cpu', GPU_CAPACITY=1024):
    # get number of points
    n = x.size()[0]

    # get dimension of data
    rows = x.size()[2]
    cols = x.size()[3]
    channels = x.size()[1]

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        scores_smoothed = np.zeros((len(scores_list), n, num_of_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n, num_of_classes))
    else:
        smooth_outputs = np.zeros((n, num_of_classes))

    image_index = -1
    print("Evaluate predictions:")
    for j in tqdm(range(num_of_batches)):

        # get inputs of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        curr_batch_size = inputs.size()[0]

        if base:
            noises_test_base = torch.empty((curr_batch_size, channels, rows, cols))
            # get relevant noises for this batch
            for k in range(curr_batch_size):
                image_index = image_index + 1
                torch.manual_seed(indices[image_index])
                noises_test_base[k:(k + 1)] = torch.randn((1, channels, rows, cols)) * sigma_smooth

            noisy_points = inputs.to(device) + noises_test_base.to(device)
        else:
            noises_test = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
            # get relevant noises for this batch
            for k in range(curr_batch_size):
                image_index = image_index + 1
                torch.manual_seed(indices[image_index])
                noises_test[(k * n_smooth):(k + 1) * n_smooth] = torch.randn(
                    (n_smooth, channels, rows, cols)) * sigma_smooth

            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # add noise to points
            noisy_points = x_tmp + noises_test.to(device)

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            # get smoothed classifier outputs
            smooth_outputs[(j * batch_size):((j + 1) * batch_size)] = noisy_outputs.reshape(-1, n_smooth, noisy_outputs.shape[1]).mean(axis=1)
            # get smoothed scores for each for all points in batch
            batch_uniform = uniform_variables[(j * batch_size):((j + 1) * batch_size)]
            batch_uniform = np.repeat(batch_uniform, n_smooth)
            for p, score_func in enumerate(scores_list):
                # get scores for all noisy outputs for all classes
                noisy_scores = score_func(noisy_outputs, np.arange(num_of_classes), batch_uniform, all_combinations=True)
                # average n_smooth scores for eac points
                smoothed_scores[p, (j * batch_size):((j + 1) * batch_size)] = noisy_scores.reshape(-1, n_smooth, noisy_scores.shape[1]).mean(axis=1)

            # clean
            del batch_uniform, noisy_scores
            gc.collect()

            # for k in range(inputs.size()[0]):
            #
            #     # get all the noisy outputs of a specific point
            #     point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]
            #
            #     # get smoothed classifier output of this point
            #     smooth_outputs2[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)
            #
            #     # generate random variable for inverse quantile score
            #     u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]
            #
            #     # run over all scores functions and compute smoothed scores with all lables
            #     for p, score_func in enumerate(scores_list):
            #         smoothed_scores2[p, ((j * batch_size) + k), :] = np.mean(
            #             score_func(point_outputs, np.arange(num_of_classes), u, all_combinations=True), axis=0)
            #     del u
            #     gc.collect()

        if base:
            del noisy_points, noisy_outputs, noises_test_base
        else:
            del noisy_points, noisy_outputs, noises_test, tmp
        gc.collect()

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
        else:
            scores_smoothed[p, :, :] = score_func(smooth_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)

    # return relevant scores
    if base:
        return scores_simple
    else:
        return smoothed_scores, scores_smoothed

def calibration(scores_simple=None, scores_smoothed=None, smoothed_scores=None, alpha=0.1, num_of_scores=2, correction=0, base=False):
    # size of the calibration set
    if base:
        n_calib = scores_simple.shape[1]
    else:
        n_calib = scores_smoothed.shape[1]

    # create container for the calibration thresholds
    thresholds = np.zeros((num_of_scores, 3))

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    bounds = np.zeros((num_of_scores, 2))
    for p in range(num_of_scores):
        if base:
            thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        else:
            thresholds[p, 1] = mquantiles(scores_smoothed[p, :], prob=level_adjusted)
            thresholds[p, 2] = mquantiles(smoothed_scores[p, :], prob=level_adjusted)

            # calculate lower and upper bounds of correction of smoothed score
            upper_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)+correction, loc=0, scale=1)
            lower_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)-correction, loc=0, scale=1)

            bounds[p, 0] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= lower_thresh])/np.size(smoothed_scores[p, :])
            bounds[p, 1] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= upper_thresh]) / np.size(smoothed_scores[p, :])

    return thresholds, bounds

def prediction(scores_simple=None, scores_smoothed=None, smoothed_scores=None, num_of_scores=2, thresholds=None, correction=0, base=False):
    # get number of points
    if base:
        n = scores_simple.shape[1]
    else:
        n = scores_smoothed.shape[1]

    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(num_of_scores):
        if base:
            S_hat_simple = [np.where(scores_simple[p, i, :] <= thresholds[p, 0])[0] for i in range(n)]
            predicted_sets.append(S_hat_simple)
        else:
            S_hat_smoothed = [np.where(scores_smoothed[p, i, :] <= thresholds[p, 1])[0] for i in range(n)]
            smoothed_S_hat = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat_corrected = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) - correction <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]

            tmp_list = [S_hat_smoothed, smoothed_S_hat, smoothed_S_hat_corrected]
            predicted_sets.append(tmp_list)

    # return predictions sets
    return predicted_sets