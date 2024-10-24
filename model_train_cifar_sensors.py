import numpy as np
import torch
from model.model_single import NEURAL_single
from model.model_single import Net_cifar
from dataset.dataset import Cifar10
import time
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model.resnet import resnet56
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr_rate = 0.01
batch_size = 200
n_iters = 180
noise_sd = 0.25

sensor_id = 18

W = 5

# airplane : 0
# automobile : 1
# bird : 2
# cat : 3
# deer : 4
# dog : 5
# frog : 6
# horse : 7
# ship : 8
# truck : 9
# is_animal: 10
# is_transportation: 11
# in_sky_or_on_land: 12
# in_water: 13
# only_on_land: 14
# four_legs: 15
# two_legs: 16
# have_wheels: 17
# no_legs_wheels: 18



label_mapping = np.array([
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]])



print('[Data] Preparing .... ')
data = Cifar10(batch_size=batch_size)
data.data_set_up(istrain=True)
data.greeting()


transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),])

trainset = torchvision.datasets.CIFAR10(root='conformal_prediction/data/', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


print('[Data] Done .... ')

print('[Model] Preparing .... ')
# model = Net_cifar()
# model = NEURAL_single(n_class=10, n_channel=3)
model = resnet56(n_class=1)
model = model.cuda()
print('[Model] Done .... ')

# loss_f = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay = 1e-4)

optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=False)
loss_f = F.cross_entropy

lr_scheduler= optim.lr_scheduler.MultiStepLR(optimizer, milestones=[91, 137], gamma=0.1)

label_mapping = label_mapping[:,sensor_id]
print(sensor_id)
print(label_mapping)

print('[Training] Starting ...')
for i in tqdm(range(n_iters)):
    for X,GT in trainloader:
        # X, GT = data.random_train_batch()
        X = X.cuda()
        X = X + torch.randn_like(X).cuda() * noise_sd


        GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
        GT = GT.cuda()
        Y = model(X)
        weight = torch.ones(GT.shape)
        weight[GT == 1] = W
        weight = weight.cuda()

        # loss = loss_f(Y,GT)
        loss = torch.mean(CrossEntropyLoss(reduction='none')(Y, GT) * weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()

    if (i + 1) % 3 == 0:
        print(f'loss at iteration {i}: {loss.item()}')

    if (i + 1) % 3 == 0:
        print('### Eval ###')
        model.eval()

        correct = 0
        total = 0
        X, GT = data.sequential_test_batch()
        while X is not None:

        # for X, GT in testloader:
            X = X.cuda()
            # X = X + torch.randn_like(X).cuda() * noise_sd
            GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
            GT = GT.cuda()
            Y = model(X)
            _, predicted = torch.max(Y.data, 1)
            correct += (predicted == GT).sum().item()
            total += len(GT)
            X, GT = data.sequential_test_batch()
        print(f'Iteration {i}: acc: {correct / total}')
        model.train()

torch.save(model.cpu(), f'pretrained_models/cifar10/model_sensor_{sensor_id}_cifar10_{noise_sd}')