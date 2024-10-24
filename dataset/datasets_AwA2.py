from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from typing import *
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
import json
import numpy as np
from scipy.io import loadmat 
from PIL import Image
# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "../data/ILSVRC2012/"

# list of all datasets
DATASETS = ["AWA","imagenet","word50_word","word50_letter", "stop_sign"]


def get_dataset(dataset: str, split: str, attribute: int = -1, hierarchy: int = -1, cls_id: list =[]) -> Dataset:
    """Return the dataset as a PyTorch Dataset object
    -1 means we do not train that knowledge predictor"""

    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "AWA":
        return _AWA(split, attribute, hierarchy)
    elif dataset == "tiny_imagenet":
        return _tiny_imagenet(split, attribute, hierarchy)
    elif dataset == "word50_letter":
        return _word50_letter(split)
    elif dataset == "word50_word":
        return _word50_word(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "mnist":
        return _mnist(split)
    elif dataset == "stop_sign":
        return _stop_sign(split, attribute)
    
def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "AWA":
        return 50
    elif dataset == "word50_letter":
        return 26
    elif dataset == "word50_word":
        return 50
    elif dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10
    elif dataset == "stop_sign":
        return 12
    
def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "AWA":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "word50_letter":
        return NormalizeLayer(_WORD50_MEAN, _WORD50_STDDEV)   
    elif dataset == "word50_word":
        return NormalizeLayer(_WORD50_MEAN, _WORD50_STDDEV)    
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)    
    elif dataset == "mnist":
        return torch.nn.Identity()
    elif dataset == "stop_sign":
        return NormalizeLayer(_STOP_MEAN, _STOP_STDDEV)    

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]
_WORD50_MEAN = [0.3710,]
_WORD50_STDDEV = [0.2058,]
_STOP_MEAN = [0.3619, 0.3306, 0.3378]
_STOP_STDDEV = [0.2833, 0.2712, 0.2759]

def _mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.MNIST("./dataset_cache", train=False, transform=transforms.ToTensor())
    
def _AWA(split: str, attribute: int = -1, hierarchy: int = -1) -> Dataset:
    tar_dir = "/data/common/AwA2/Animals_with_Attributes2"
    if split == "train":
        subdir = os.path.join(tar_dir, "train")
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(tar_dir, "test")
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    if attribute != -1:
        # prepare the dataset for attribute training
        return AttribubteFolder(subdir, transform, attribute)
    elif hierarchy != -1:
        # prepare the dataset for hierarchy training
        return HierarchyFolder(subdir, transform, hierarchy)
    else:
        # prepare the dataset for main predictor training
        return ImageFolder(subdir, transform)

def _stop_sign(split: str, attribute) -> Dataset:
    tar_dir = "../data/stop_sign/"

    x = np.load(tar_dir+f"raw_feature_{split}.npy")
    y = np.load(tar_dir+f"raw_label_{split}.npy")
    x = torch.tensor(x).float().permute(0,3,1,2)/255
    y = torch.tensor(y).long()
    return StopSignFolder(x,y,attribute)
    
def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())
    
def _imagenet(split: str) -> Dataset:
#     if not IMAGENET_LOC_ENV in os.environ:
#         raise RuntimeError("environment variable for ImageNet directory not set")

#     dir = os.environ[IMAGENET_LOC_ENV]
    dir = "../data/ILSVRC2012/"
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    
def _word50_letter(split: str) -> Dataset:
    dir = "../data/word50/word50.mat"
    word50 = loadmat(dir)
    if split == "train":
        data = word50['train_feat']
        data = torch.FloatTensor(data).T.reshape(-1,1,28,28)
        label = word50['train_label']
        label = torch.LongTensor(label).squeeze()
    elif split == "test":
        data = word50['test_feat']
        data = torch.FloatTensor(data).T.reshape(-1,1,28,28)
        label = word50['test_label']
        label = torch.LongTensor(label).squeeze()
    return Word50CFolder(data, label)

def _word50_word(split: str) -> Dataset:

    """
    Concatenate the 5 character into a word, and label it.
    """
    tar_dir = "../data/word50/word50_main.pt"
    word50 = torch.load(tar_dir)
    if split == "train":
        data = word50['train_all_feat'].reshape(-1,1,5*28,28)
        label = word50['train_all_label']
    elif split == "test":
        data = word50['test_all_feat'].reshape(-1,1,5*28,28)
        label = word50['test_all_label']

    return Word50WFolder(data, label)

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
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds

class AttribubteFolder(ImageFolder):
    """
        Get the AwA for corresponding attribute group.
        like attribute = o => train the attribute black
        ...
    """
    def __init__(self, root, transform=None, attribute = 0):
        super(AttribubteFolder, self).__init__(root, transform)
        self.indices = range(len(self)) 
        self.animal_classes = pd.read_table('../data/Animals_with_Attributes2/classes.txt',sep='\t',header=None) 
        self.predicate_matrix = pd.read_table('../data/Animals_with_Attributes2/predicate-matrix-binary.txt',sep=' ',header=None) 
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.attribute = attribute
        self.attribute_matrix = self.get_attr_matrix()
        
    def __getitem__(self, index):

        path = self.imgs[index][0] 
        label = self.imgs[index][1]
        label = self.anilabel2attribute(label)
        
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img,label
    
    def samples_weights(self):
        # balance the positive example and negative example
        # balance the number of images from different animal classes
        labels = pd.value_counts(self.targets).sort_index()
        weights = 1./ torch.tensor(labels.to_list(), dtype=torch.float)
        attr_col = self.attribute_matrix[:,self.attribute].view(-1)
        attr_sum = (attr_col == 1).sum()
        attr_weight = torch.ones(weights.shape)
        attr_weight[attr_col == 0] = attr_sum/(50.0-attr_sum)
        weights *= attr_weight
        samples_weights = weights[self.targets]
        return samples_weights
    
    def anilabel2attribute(self, label):
        animal = self.idx_to_class[label]
        mapping_index = self.animal_classes[self.animal_classes[1] == animal].index[0]
        all_arrtribute = self.predicate_matrix.iloc[mapping_index].to_list()
        part_attribute = all_arrtribute[self.attribute]
        return torch.FloatTensor([part_attribute])
    
    def get_attr_matrix(self):
        attr_matrix = torch.FloatTensor([])
        for label in range(50):
            animal = self.idx_to_class[label]
            mapping_index = self.animal_classes[self.animal_classes[1] == animal].index[0]
            all_arrtribute = self.predicate_matrix.iloc[mapping_index].to_list()
            attr_matrix = torch.cat((attr_matrix,torch.FloatTensor(all_arrtribute).unsqueeze(0)),dim=0)
        return attr_matrix

    
class HierarchyFolder(ImageFolder):
    """
        Get the ImageFolder for corresponding hierarchy group.
        like hierarchy = 0 => train the classificaction of the leaf node persian cat.
    """
    def __init__(self, root, transform=None, hierarchy = 0):
        super(HierarchyFolder, self).__init__(root, transform)
        self.indices = range(len(self)) 
        # map the leaf node to the label.
        with open('../data/Animals_with_Attributes2/hierarchy_training.json', 'r') as f:
            self.hierarchy_training = json.load(fp=f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.hierarchy = hierarchy
        self.hierarchy_matrix = self.get_hier_matrix()

    def __getitem__(self, index):

        path = self.imgs[index][0] 
        label = self.imgs[index][1]
        label = torch.FloatTensor([self.anilabel2hierarchy(label)])
        
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img,label
    
    def samples_weights(self):
        # balance the positive example and negative example
        # balance the number of images from different animal classes
        labels = pd.value_counts(self.targets).sort_index()
        weights = 1./ torch.tensor(labels.to_list(), dtype=torch.float)
        hier_col = self.hierarchy_matrix[:,self.hierarchy].view(-1)
        hier_sum = (hier_col == 1).sum()
        hier_weight = torch.ones(weights.shape)
        hier_weight[hier_col == 0] = hier_sum/(50.0-hier_sum)
        weights *= hier_weight
        samples_weights = weights[self.targets]
        return samples_weights
    
    def anilabel2hierarchy(self, label):
        animal = self.idx_to_class[label]
        if animal in self.hierarchy_training[str(self.hierarchy)]:
            return 1
        return 0

    def get_hier_matrix(self):
        hier_matrix = torch.zeros((50,len(self.hierarchy_training)))
        for rv in range(len(self.hierarchy_training)):
            for cls in self.hierarchy_training[str(rv)]:
                hier_matrix[self.class_to_idx[cls],rv] = 1
        return hier_matrix

    def pos_num(self):
#         if len(self.positive) > 100:
#              return min(30000,len(torch.nonzero(torch.IntTensor(self.targets), as_tuple=False)))
#         return min(25000,len(torch.nonzero(torch.IntTensor(self.targets), as_tuple=False)))
        return 5000 

class Word50CFolder(Dataset):
    '''
    classification on the character level.
    '''
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]  
        y = self.label[index] 
        return x,y 

    def samples_weights(self):
        # balance the number of different characters
        labels = pd.value_counts(self.label.numpy()).sort_index()
        weights = 1./ torch.tensor(labels.to_list(), dtype=torch.float)
        tmp = torch.zeros(26)
        tmp[:23] = weights[:23]
        tmp[-2:] = weights[-2:]
        weights = tmp
        samples_weights = weights[self.label]
        return samples_weights
    
    def __len__(self):
        return self.data.shape[0] 
    
class Word50WFolder(Dataset):
    '''
    classification on the word level.
    '''
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]  
        y = self.label[index] 
        return x,y 

    def samples_weights(self):
        # balance the number of different words
        labels = pd.value_counts(self.label.numpy()).sort_index()
        weights = 1./ torch.tensor(labels.to_list(), dtype=torch.float)
        samples_weights = weights[self.label]
        return samples_weights
    
    def __len__(self):
        return self.data.shape[0] 

    
class StopSignFolder(Dataset):
    '''
    classification on the word level.
    '''
    def __init__(self, data, label, attribute = -1):
        self.data = data
        self.original_label = label
        self.label = label
        self.attribute = attribute
        self.gt_matrix = torch.load("../data/stop_sign/stop_sign_gt_matrix.pt").long()
        if self.attribute != -1 :
            self.label = self.gt_matrix[:,self.attribute][self.label]
        
    def __getitem__(self, index):
        x = self.data[index]  
        y = self.label[index] 
        return x,y 

    def samples_weights(self):
        # balance the number of different classes
        if self.attribute == -1:
            labels = pd.value_counts(self.label.numpy()).sort_index()
            weights = 1./ torch.tensor(labels.to_list(), dtype=torch.float)
            samples_weights = weights[self.label]
        else:
            ori_labels = pd.value_counts(self.original_label.numpy()).sort_index()
            ori_weights = 1./ torch.tensor(ori_labels.to_list(), dtype=torch.float)
            ori_samples_weights = ori_weights[self.original_label]
            
            ratio = torch.tensor([self.gt_matrix[:,self.attribute].sum(),12-self.gt_matrix[:,self.attribute].sum()]).float()
            samples_weights =ratio[self.label]
            samples_weights *= ori_samples_weights
            
        return samples_weights
    
    def __len__(self):
        return self.data.shape[0] 
        
    def pos_num(self):
#         if self.attribute == -1:
        return self.data.shape[0] 
#         else:
#             minmum = min(self.label.sum(),(1-self.label).sum())
#             return max(2 * minmum, 10000)
