from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler, \
random_split, TensorDataset
from torch import Tensor
#from torch.utils.data import random_split, Tensor
from PIL import Image
import torch, os
import numpy as np
import re
import os
from numpy.random import standard_normal, binomial, choice
from numpy import sqrt
from my_datahanddlers import map_to
import pandas as pd
#from torch.nn import Sequential
#from torch.jit import script as jscript

class LoadDataset():
    def __init__(self, input_dim, target_dim=None, batch_size_train=None, batch_size_test=None, seed=42):
        self.input_dim = input_dim
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.seed = seed

        #To normalize the input images data.
        mean = [.485, .456, .406]
        std  = [.229, .224, .225]
#        mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
#        std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

        # Note that we apply data augmentation in the training dataset.
        #You can change as you want.
        #removed this data transformation, opted for my own method
        self.transformations_train = transforms.Compose([
            transforms.Resize(input_dim),
            transforms.CenterCrop(input_dim),
            transforms.ToTensor(),
            transforms.RandomApply([
               # transforms.RandomChoice([
                    transforms.ColorJitter(
                            brightness=.5,
                            contrast=.5,
                            saturation=.5,
                            hue=.25,
                    ),
                    #os que seguem abaixo, se desagradar tirar
                    #transforms.GaussianBlur(15,sigma=(.1,1.5)),
                    #transforms.ElasticTransform(alpha=50.),
                    #dps acrescentar pepper-and-salt noise
                 ], p=.2,
            ),
            transforms.Normalize(mean = mean, std = std),
        ])

        # Note that we do not apply data augmentation in the test dataset.
        self.transformations_test = transforms.Compose([
            transforms.Resize(input_dim),
            transforms.CenterCrop(input_dim),
            transforms.ToTensor(),              #not a scriptable transformation
            transforms.Normalize(mean = mean, std = std),
        ])

        target_dim = target_dim or input_dim
        self.transformations_target = transforms.Compose([
            transforms.Resize(target_dim),
            transforms.CenterCrop(target_dim),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255),
            transforms.Lambda(lambda x: x.type(torch.long)),
            transforms.Lambda(lambda x: torch.where(x == 255, 21, x)),      #null channel is the 22th
        ])

#   def get_indices(self, dataset, split_ratio):
#       nr_samples = len(dataset)
#       indices = list(range(nr_samples))
#       train_size = nr_samples - int(np.floor(split_ratio * nr_samples))
#       np.random.shuffle(indices)
#       train_idx, test_idx = indices[:train_size], indices[train_size:]
#       return train_idx, test_idx

#   def get_dataset(self, root_path, dataset_name, split_ratio, idx_path=None, idx=None):
#       self.dataset_name = dataset_name
#       self.idx_path = idx_path
#       self.idxs = idx
#       def func_not_found():
#           print(f'No dataset {self.dataset_name} is found')


#       func_name = getattr(self, self.dataset_name, func_not_found)
#       train_loader, val_loader, test_loader = func_name(root_path, split_ratio)
#       return train_loader, val_loader, test_loader

    def voc_seg(self, root_path):
        # This method loads Cifar-10 dataset.
        # saves the seed
        torch.manual_seed(self.seed)

        # This downloads the training and test CIFAR-10 datasets and also applies transformation  in the data.
        try:
            train_set = datasets.VOCSegmentation(root=root_path, image_set='train',
                                                transform=self.transformations_train, target_transform=self.transformations_target)
        except:
            train_set = datasets.VOCSegmentation(root=root_path, image_set='train', download=True,
                                                transform=self.transformations_train, target_transform=self.transformations_target)
        try:
            test_val_set = datasets.VOCSegmentation(root=root_path, image_set='val',
                                                transform=self.transformations_test, target_transform=self.transformations_target)
        except:
            test_val_set = datasets.VOCSegmentation(root=root_path, image_set='val', download=True,
                                                transform=self.transformations_test, target_transform=self.transformations_target)
#        try:
#            test_set = datasets.VOCSegmentation(root=root_path, image_set='val',
#                                                transform=self.transformations_test, target_transform=self.transformations_target)
#        except:
#            test_set = datasets.VOCSegmentation(root=root_path, image_set='val', download=True,
#                                                transform=self.transformations_test, target_transform=self.transformations_target)

        val_size = int(.4*len(test_val_set))
        test_size = int(len(test_val_set) - val_size)

        val_set, test_set = random_split(test_val_set, [val_size, test_size])

        if not self.batch_size_train:
            return train_set, val_set, test_set

        #This block creates data loaders for training, validation and test datasets.
        train_loader = DataLoader(train_set, self.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, self.batch_size_test, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, self.batch_size_test, num_workers=4, pin_memory=True)

        return train_loader, val_loader, test_loader

    def get_dataset(self, root_path, dataset_name):#, split_ratio, idx_path=None, idx=None):
        self.dataset_name = dataset_name
#        self.idx_path = idx_path
#        self.idxs = idx
        def func_not_found():
            print(f'No dataset {self.dataset_name} is found')


        func_name = getattr(self, self.dataset_name, func_not_found)
        train_loader, val_loader, test_loader = func_name(root_path)#, split_ratio)
        return train_loader, val_loader, test_loader
