import numpy as np
import torch
from model.model import NEURAL
from dataset.dataset import DataMain
import time
import os

batch_size = 400

def test(data, model, label_mapping, noise_sd):
    correct = 0
    tot = 0

    X, GT = data.sequential_test_batch()
    while X is not None:

        X = X.cuda()  # to(device)
        X = X + torch.randn_like(X).cuda() * noise_sd
        GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()

        Y = model(X)
        Y = torch.argmax(Y, dim=1)

        this_batch_size = len(Y)

        for i in range(this_batch_size):
            tot += 1

            if GT[i] == Y[i]:
                correct += 1

        X, GT = data.sequential_test_batch()
    acc = 100*correct / tot
    return acc


print('[Data] Preparing .... ')
data = DataMain(batch_size=batch_size)
data.data_set_up(istrain=False)
data.greeting()
print('[Data] Done .... ')


for sigma in [0.12]:
    print("sigma = %.2f" % sigma)
    print("Hierarchy => Main")

    mappings = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]).astype(np.int64)

    for i in range(12):
        model = NEURAL(n_class=1, n_channel=3)
        if os.path.exists("pretrained_models/hier/model_%d_%.2f_5.pt" % (i, sigma)):
            model.load_state_dict(torch.load("pretrained_models/hier/model_%d_%.2f_5.pt" % (i, sigma)))
        else:
            model.load_state_dict(torch.load("pretrained_models/hier/model_%d_%.2f.pt" % (i, sigma)))
        model = model.cuda()
        model.eval()

        label_mapping = mappings[i]
        acc = test(data, model, label_mapping, sigma)
        print(f"test accuracy of sensor of class {i}: {acc}")

    mappings = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]).astype(np.int64)

    print("Main => Attribute")

    for i in range(13):
        model = NEURAL(n_class=1, n_channel=3)
        if os.path.exists("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, sigma)):
            model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f_5.pt" % (i, sigma)))
        else:
            model.load_state_dict(torch.load("pretrained_models/attr/model_%d_%.2f.pt" % (i, sigma)))
        model = model.cuda()
        model.eval()

        label_mapping = mappings[i]
        acc = test(data, model, label_mapping, sigma)

        print(f"test accuracy of sensor of attribute {i}: {acc}")