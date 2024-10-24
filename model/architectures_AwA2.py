import torch
import torch.nn as nn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.lenet import LeNet
from archs.mlp import MLP
from archs.awa_resnet import AWADNN
from archs.neural import NEURAL
from dataset.datasets_AwA2 import get_normalize_layer

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110",'neural','lenet', 'MLP']

def get_architecture(arch: str, dataset: str, classes: int = 50) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if dataset == "AWA":
        if arch == 'catmlp':
            model = torch.nn.DataParallel(AWADNN(classes = 50+85+28)).cuda()
        elif arch == 'addmlp':
            return MLP().cuda()
        else:
            model = torch.nn.DataParallel(AWADNN(classes = classes)).cuda()
    elif dataset == "word50_letter":
        model = torch.nn.DataParallel(MLP(n_class = 26)).cuda()
    elif dataset == "word50_word":
        model = torch.nn.DataParallel(MLP(n_class = 50)).cuda()
    elif arch == "cifar_resnet20":
        model = torch.nn.DataParallel(resnet_cifar(depth=20, num_classes=10)).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "cifar_resnet110_attribute":
        model = resnet_cifar(depth=110, num_classes=2).cuda()
    elif arch == "lenet":
        model = LeNet(num_classes=10).cuda()
    elif arch == "lenet_attribute":
        model = LeNet(num_classes=1).cuda()
    elif dataset == "stop_sign":
        if arch == "neural":
            model = NEURAL(12, 3).cuda()
        elif arch == "neural_attribute":
            model = NEURAL(2, 3).cuda()
        elif arch == 'catmlp':
            model = NEURAL(12+20, 3).cuda()
        elif arch == 'addmlp':
            return MLP(32,32).cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)

class MLP(nn.Module):
    def __init__(self, input_dim = 50+85+28, classes = 50+85+28):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(input_dim, classes),
        )

    def forward(self, x):
        return self.layers(x)