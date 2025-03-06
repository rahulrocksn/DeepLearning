# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:08:43 2022

@author: Mir Imtiaz Mostafiz

Contain all the utility functions
"""
import argparse
import sys
import os
import logging
import torch
import numpy as np
import random
import mnist as mnist
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

data_folder = None
n_training_examples = None
n_test_examples = None
gpu_id = -1
depth = "shallow"
batch_size = 100
device = None
lr = 0.01
epochs = 10
thrng = torch.Generator("cpu")
num_samples = 10
std_dev = 1
delta = 0.05
num_leapfrog = 10
is_binary_class = False
loaded = False
p_num_samples = 20
p_num_leapfrog = 10
num_bins = 10

def bin_config(get_arg_func):
    # get arguments
    args = get_arg_func(sys.argv[1:])

    # set logger
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    formatter = logging.Formatter('[%(levelname)s][%(name)s] %(message)s')
    try:
        # if output_folder is specified in the arguments
        # put the log in there
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)
        fpath = os.path.join(args.output_folder, 'log')
    except:
        # otherwise, create a log file locally
        fpath = 'log'
    fileHandler = logging.FileHandler(fpath)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return args


def get_calibration_points(y_labels, y_scores_all_prob, num_bins):
    """
    Calculates confidences, accuracies, bin counts and bin indices for reliablity diagram

    Parameters
    ----------
    y_labels : torch.tensor of shape(num_examples), integers
        true labels.
    y_scores_all_prob : torch.tensor of shape(num_examples, num_class), float32
        probability of classes.
    num_bins : integer
        number of bins.
    
    Returns
    -------
    confidences : torch.tensor of shape(num_bins), float32
        average probabilities per bin.
    accuracies : torch.tensor of shape(num_bins), float32
        average accuracies per bin.
    bin_indices : torch.tensor of shape(num_examples), int
        indicates which example's probability will go to which bin.
    bin_counts : torch.tensor of shape(num_bins), int
        counts per bin.

    """
    max_probs_values_indices = torch.max(y_scores_all_prob,1)
    max_probs = max_probs_values_indices[0].cpu()
    predicted_labels = max_probs_values_indices[1].cpu()
    
    confidences = torch.zeros((num_bins)).cpu()
    accuracies = torch.zeros((num_bins)).cpu()
    
    bin_size = 1.0/ num_bins
    bin_indices = (torch.floor(max_probs/bin_size)-1).long().cpu()
    
    confidences.index_add_(0, bin_indices, max_probs)
    
    proper_predictions = torch.zeros(y_labels.shape[0]).cpu()
    proper_predictions[ y_labels.cpu() == predicted_labels ] = 1
    
    accuracies.index_add_(0, bin_indices, proper_predictions)   
    bin_counts = torch.bincount(bin_indices).float().cpu()
    
    mask = (bin_counts !=0).cpu()
    
    confidences[mask] /= bin_counts[mask]
    accuracies[mask] /= bin_counts[mask]
    
    
    return confidences, accuracies, bin_indices, bin_counts

def get_expected_calibration_error(confidences, accuracies, bin_counts):
    """
    Returns Expected Calibration Error

    Parameters
    ----------
    confidences : torch.tensor of shape(num_bins), float32
        average probabilities per bin.
    accuracies : torch.tensor of shape(num_bins), float32
        average accuracies per bin.
    bin_counts : torch.tensor of shape(num_bins), int
        counts per bin.

    Returns
    -------
    float
        Expected calibration error.

    """
    
    return (torch.sum(torch.abs(confidences-accuracies)*bin_counts)/(torch.sum(bin_counts))).item()

def get_expected_calibration_error_threshold(confidences, accuracies, bin_counts, num_bins, threshold):
    """
    Get expected calibration error over a threshold

    Parameters
    ----------
    confidences : torch.tensor of shape(num_bins), float32
        average probabilities per bin.
    accuracies : torch.tensor of shape(num_bins), float32
        average accuracies per bin.
    bin_indices : torch.tensor of shape(num_examples), int
        indicates which example's probability will go to which bin.
    bin_counts : torch.tensor of shape(num_bins), int
        counts per bin.
    threshold : float
        probability value threshold for better-than-odds model.

    Returns
    -------
    Float
        Expected Calibration error over a threshold.

    """
    
    idx = torch.arange(num_bins).cpu()
    mask = torch.zeros_like(idx).cpu()
    mask[idx>=(threshold*num_bins-1)] = 1
    return (torch.sum(torch.abs(confidences-accuracies)*bin_counts*mask)/(torch.sum(bin_counts))).item()

    

def draw_reliablity_diagram(confidences, accuracies, path):
    """
    Draws Reliability Diagram

    Parameters
    ----------
    confidences : torch.tensor of shape(num_bins), float32
        average probabilities per bin.
    accuracies : torch.tensor of shape(num_bins), float32
        average accuracies per bin.
    path : String
        Path to save the plot.

    Returns
    -------
    None.

    """
    plt.figure()

    plt.plot(confidences.detach().numpy(), accuracies.detach().numpy(),label="reliablity diagram", marker = 'o')
    plt.title(path)
    plt.ylabel('Accuracies [0,1]')
    plt.xlabel('Confidences [0,1]')
    plt.legend(loc=4)
    plt.savefig(path)
    plt.show()


def plot_graph(x, y, xtitle, ytitle, label, path):
    """
    
    Plots a graph
    
    Parameters
    ----------
    x : numpy array
        x of graph.
    y : numpy array
        y of graph.
    xtitle : String
        legend of x-axis.
    ytitle : String
        legend of x-axis.
    label : String
        Label of graph
    path : String
        Path to save the plot.

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(x,y,label=label, marker = 'o')
    plt.title(path)
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    plt.legend(loc=4)
    plt.savefig(path)
    plt.show()
    

    
    
    
def analyse_calibration(y_labels, y_scores_all_prob, num_bins, path, threshold = 0.5):
    """
    Calculates confidences, accuracies, bin counts, Expected calibration error (general and with threshold) for reliability diagram

    Parameters
    ----------
    y_labels : torch.tensor of shape(num_examples), integers
        true labels.
    y_scores_all_prob : torch.tensor of shape(num_examples, num_class), float32
        probability of classes.
    num_bins : integer
        number of bins.
    path : String
        path to save.
    threshold : float, optional
        Threshold for calculating better-than-odds model's ECE. The default is 0.5.

    Returns
    -------
    ece : float
        Expected Calibration Error.
    ece_threshold : float
        Expected Calibration Error over a threshold.

    """
    
    confidences, accuracies, bin_indices, bin_counts = get_calibration_points(y_labels, y_scores_all_prob, num_bins)

    ece = get_expected_calibration_error(confidences, accuracies, bin_counts)
    ece_threshold = get_expected_calibration_error_threshold(confidences, accuracies, bin_counts, num_bins, threshold)
    draw_reliablity_diagram(confidences, accuracies, path)
    
    print("Expected Calibration Error: ", ece)
    print("Expected Calibration Error over threshold " + str(threshold*100) + "\%: ", ece_threshold)
    
    return ece, ece_threshold
    
        
        
    
def plot_roc_auc_curve(y_labels, y_scores_all_prob, path):
    """
    Plots ROC curve and calculate AUC

    Parameters
    ----------
    y_labels : torch.tensor of shape (n_examples, )
        gold labels/classes .
    y_scores_all_prob : torch.tensor of shape (n_examples, n_classes )
        predicted probabilities of classes per example.
    path : String
        path to save the plot.

    Returns
    -------
    fpr : float32 list
        false positive rate under different thresholds.
    tpr : float32 list
        true positive rate under different thresholds.
    threshold : float32 list
        thresholds (probability bounds) to denote something as positive example (label 1).
    auc : float32
        area under ROC.

    """
    y_labels = y_labels.cpu().detach().numpy()
    y_scores = y_scores_all_prob[:, 1].cpu().detach().numpy()
    fpr, tpr, threshold = metrics.roc_curve(y_labels, y_scores)
    
    auc = metrics.roc_auc_score(y_labels, y_scores)

    plt.figure()
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.title(path)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig(path)
    plt.show()

    
    return fpr, tpr, threshold, auc
    
def accuracy(gold, pred):
    """
    Return the accuracy from gold labels and predicted labels

    Parameters
    ----------
    gold : torch.tensor of shape (n_examples,)
        real/gold labels of examples.
    pred : torch.tensor of shape (n_examples,)
        predicted labels of examples.

    Returns
    -------
    ret : float32
        ratio of correctly predicted labels and total labels

    """
    try:
        denom = gold.shape[0]
        nom = (gold.squeeze().long() == pred).sum()
        ret = float(nom) / denom
    except:
        denom = gold.data.shape[0]
        nom = (gold.data.squeeze().long() == pred.data).sum()
        ret = float(nom) / denom
    return ret

def fix_seeds(seed_value):
    """
    

    Parameters
    ----------
    seed_value : Integer
        Fix torch, numpy and python random number generator seeds.

    Returns
    -------
    None.

    """
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    thrng.manual_seed(seed_value)

def load_mnist(data_folder, max_n_examples_train = -1,  max_n_examples_test = -1, is_binary = False):
    """
    

    Parameters
    ----------
    data_folder : String
        data folder name.
    max_n_examples_train : Integer, optional
        number of training examples to load. The default is -1.
    max_n_examples_test : Integer, optional
        number of test examples to load. The default is -1.
    is_binary: boolean
        if true, then converts odd numbers to 1 and even to 0.
        if false, keep them as it is

    Returns
    -------
    X_train : numpy.ndarray of shape (max_n_examples_train, 28, 28), type uint8, max: 255, min: 0
        training images.
    y_train : numpy.ndarray of shape (max_n_examples_train, ), type int64, max: 9, min: 0
        training labels.
    X_test : Tnumpy.ndarray of shape (max_n_examples_test, 28, 28), type uint8, max: 255, min: 0
        test images.
    y_test : numpy.ndarray of shape (max_n_examples_test, ), type int64, max: 9, min: 0
        test labels..

    """
    X_train, y_train = mnist.load_train_data(data_folder, max_n_examples_train)
    X_test, y_test = mnist.load_test_data(data_folder, max_n_examples_test)
    
    if is_binary:
        
        odd_idx_train = (y_train % 2 == 1)
        odd_idx_test = (y_test % 2 == 1)
        even_idx_train = (y_train % 2 == 0)
        even_idx_test = (y_test % 2 == 0)
        
        y_train[odd_idx_train] = 1
        y_train[even_idx_train] = 0
        y_test[even_idx_test] = 0
        y_test[odd_idx_test] = 1
        mnist.N_CLASSES = 2
        
        
        
    return X_train, y_train, X_test, y_test
    
def one_hot(y, n_classes):
    """
    converts a target vector into one hot encoding

    Parameters
    ----------
    y : numpy.ndarray of shape (n_examples_test, ), type int64, max: 9, min: 0
        labels.
    n_classes : numpy.ndarray of shape (max_n_examples_test, ), type int64, max: 9, min: 0
        number of classes.

    Returns
    -------
    y_1hot : numpy.ndarray of shape (max_n_examples_test, n_classes), type float32, max: 1., min: 0.
        one hot encoding of labels

    """
    m = y.shape[0]
    y_1hot = np.zeros((m, n_classes), dtype=np.float32)
    y_1hot[np.arange(m), np.squeeze(y)] = 1
    return y_1hot

def reshape_data(X_train, y_train, X_test, y_test):
    """
    Reshapes training and test image data to one dimensional vectors
    also converts training and test labels into one-hot encoding

    Parameters
    ----------
    X_train : numpy.ndarray of shape (max_n_examples_train, 28, 28), type uint8, max: 255, min: 0
        training images.
    y_train : numpy.ndarray of shape (max_n_examples_train, ), type int64, max: 9, min: 0
        training labels.
    X_test : Tnumpy.ndarray of shape (max_n_examples_test, 28, 28), type uint8, max: 255, min: 0
        test images.
    y_test : numpy.ndarray of shape (max_n_examples_test, ), type int64, max: 9, min: 0
        test labels..

    Returns
    -------
    X_train : numpy.ndarray of shape (max_n_examples_train, 28*28), type uint8, max: 255, min: 0
        training images reshaped.
    y_train_1hot : numpy.ndarray of shape (max_n_examples_test, n_classes), type float32, max: 1., min: 0.
        one hot encoding of training labels
    X_test : numpy.ndarray of shape (max_n_examples_test, 28*28), type uint8, max: 255, min: 0
        test images reshaped.
    y_test_1hot : numpy.ndarray of shape (max_n_examples_test, n_classes), type float32, max: 1., min: 0.
        one hot encoding of test labels

    """
    X_train = X_train.reshape((X_train.shape[0], -1))
    y_train_1hot = one_hot(y_train, mnist.N_CLASSES)
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_test_1hot = one_hot(y_test, mnist.N_CLASSES)
    
    return X_train, y_train_1hot, X_test, y_test_1hot

    

def torchify_set(X, y, y_1hot):
    """
    Converts image, label, label one-hot encoding to torch tensors

    Parameters
    ----------
    X : numpy.ndarray of shape (n_examples, 28*28), type uint8, max: 255, min: 0
        images .
    y : numpy.ndarray of shape (n_examples, ), type int64, max: 9, min: 0
        labels.
    y_1hot : numpy.ndarray of shape (n_examples, n_classes), type float32, max: 1., min: 0.
        one hot encoding of labels

    Returns
    -------
    X : torch.tensors of shape (n_examples, 28*28), type torch.float32, max: 255, min: 0
        images.
    y : torch.tensors of shape (n_examples, ), type torch.int64, max: 9, min: 0
        labels
    y_1hot : torch.tensors of shape (n_examples, n_classes), type torch.int64, max: 1., min: 0.
        one hot encoding of labels.

    """
    X, y, y_1hot = torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(y)
    X = X.type(torch.FloatTensor)
    return X, y, y_1hot

    
def torchify_tensors(X_train, y_train, y_train_1hot, X_test, y_test, y_test_1hot ):
    """
    Converts train and test image, label, label one-hot encoding to torch tensors

    Parameters
    ----------
    X_train : numpy.ndarray of shape (n_examples_train, 28*28), type uint8, max: 255, min: 0
        training images .
    y_train : numpy.ndarray of shape (n_examples_train, ), type int64, max: 9, min: 0
        training labels.
    y_train_1hot : numpy.ndarray of shape (n_examples_train, n_classes), type float32, max: 1., min: 0.
        one hot encoding of training labels
    X_test : numpy.ndarray of shape (n_examples_test, 28*28), type uint8, max: 255, min: 0
        testing images .
    y_test : numpy.ndarray of shape (n_examples_test, ), type int64, max: 9, min: 0
        testing labels.
    y_test_1hot : numpy.ndarray of shape (n_examples_test, n_classes), type float32, max: 1., min: 0.
        one hot encoding of testing labels
    
    Returns
    -------
    X_train : torch.tensors of shape (n_examples_train, 28*28), type torch.float32, max: 255, min: 0
        training images.
    y_train : torch.tensors of shape (n_examples_train, ), type torch.int64, max: 9, min: 0
        training labels
    y_train_1hot : torch.tensors of shape (n_examples_test, n_classes), type torch.int64, max: 1., min: 0.
        one hot encoding of training labels.
    X_test : torch.tensors of shape (n_examples_test, 28*28), type torch.float32, max: 255, min: 0
        testing images.
    y_test : torch.tensors of shape (n_examples_test, ), type torch.int64, max: 9, min: 0
        testing labels
    y_test_1hot : torch.tensors of shape (n_examples_test, n_classes), type torch.int64, max: 1., min: 0.
        one hot encoding of testing labels.
    
    """
    X_train, y_train, y_train_1hot = torchify_set(X_train, y_train, y_train_1hot)
    X_test, y_test, y_test_1hot = torchify_set(X_test, y_test, y_test_1hot)
    

    return X_train, y_train, y_train_1hot, X_test, y_test, y_test_1hot 

def convert_tensor_variable_gpu(tensor, requires_grad, gpu_id):
    """
    

    Parameters
    ----------
    tensor : torch.tensor
        torch.tensor to convert into torch.autograd.Variable.
    requires_grad : Boolean
        .flag to indicate if the variable is learnable
    gpu_id : integer
        gpu_id, if -1, then cpu.

    Returns
    -------
    var_tensor : torch.Autograd.Variable, if gpu_id !=-1 then it is a cuda tensor
        torch.autograd.Variable .

    """
    var_tensor = torch.autograd.Variable(tensor, requires_grad = requires_grad)
    if gpu_id != -1:
        var_tensor = var_tensor.cuda(gpu_id)
        
    return var_tensor

def convert_tensors_variable_gpu(X_train, y_train, y_train_1hot, X_test, y_test, y_test_1hot, gpu_id):
    """
    Converts training and test images, labels and one hot encodings to torch Variables and cuda tensors

    Parameters
    ----------
    X_train : torch.tensors of shape (n_examples_train, 28*28), type torch.float32, max: 255, min: 0
        training images.
    y_train : torch.tensors of shape (n_examples_train, ), type torch.int64, max: 9, min: 0
        training labels
    y_train_1hot : torch.tensors of shape (n_examples_test, n_classes), type torch.int64, max: 1., min: 0.
        one hot encoding of training labels.
    X_test : torch.tensors of shape (n_examples_test, 28*28), type torch.float32, max: 255, min: 0
        testing images.
    y_test : torch.tensors of shape (n_examples_test, ), type torch.int64, max: 9, min: 0
        testing labels
    y_test_1hot : torch.tensors of shape (n_examples_test, n_classes), type torch.int64, max: 1., min: 0.
        one hot encoding of testing labels.
    gpu_id : integer
        gpu_id, if -1, then cpu.
    
    Returns
    -------
    X_train : torch.tensors of shape (n_examples_train, 28*28), type torch.float32, max: 255, min: 0
        training images.
    y_train : torch.tensors of shape (n_examples_train, ), type torch.int64, max: 9, min: 0
        training labels
    y_train_1hot : torch.tensors of shape (n_examples_test, n_classes), type torch.int64, max: 1., min: 0.
        one hot encoding of training labels.
    X_test : torch.tensors of shape (n_examples_test, 28*28), type torch.float32, max: 255, min: 0
        testing images.
    y_test : torch.tensors of shape (n_examples_test, ), type torch.int64, max: 9, min: 0
        testing labels
    y_test_1hot : torch.tensors of shape (n_examples_test, n_classes), type torch.int64, max: 1., min: 0.
        one hot encoding of testing labels.

    """
    
    X_train = convert_tensor_variable_gpu(X_train, True, gpu_id)
    y_train = convert_tensor_variable_gpu(y_train, False, gpu_id)
    y_train_1hot = convert_tensor_variable_gpu(y_train_1hot, False, gpu_id)
    
    X_test = convert_tensor_variable_gpu(X_test, True, gpu_id)
    y_test = convert_tensor_variable_gpu(y_test, False, gpu_id)
    y_test_1hot = convert_tensor_variable_gpu(y_test_1hot, False, gpu_id)
    
    return X_train, y_train, y_train_1hot, X_test, y_test, y_test_1hot

