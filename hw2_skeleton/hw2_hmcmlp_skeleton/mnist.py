# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:05:05 2022

@author: Mir Imtiaz Mostafiz

This module is used for loading MNIST data (train and test)
"""
import os
import torch
from torchvision import datasets, transforms
import numpy as np
import logging


TRAIN_IMAGE_FILE_NAME='train-images-idx3-ubyte.gz'
TRAIN_LABEL_FILE_NAME='train-labels-idx1-ubyte.gz'
TEST_IMAGE_FILE_NAME='t10k-images-idx3-ubyte.gz'
TEST_LABEL_FILE_NAME='t10k-labels-idx1-ubyte.gz'
N_CLASSES = 10



def load_train_data(folder, max_n_examples=-1):

    train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

    if max_n_examples == -1:
        return train_data.data.numpy(), train_data.targets.numpy()
    else:
        return train_data.data.numpy()[:max_n_examples], train_data.targets.numpy()[:max_n_examples]
    

def load_test_data(folder, max_n_examples=-1):

    test_data = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())

    if max_n_examples == -1:
        return test_data.data.numpy(), test_data.targets.numpy()
    else:
        return test_data.data.numpy()[:max_n_examples], test_data.targets.numpy()[:max_n_examples]
    

