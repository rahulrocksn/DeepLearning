# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:05:54 2022
Modified on Wed Feb 19 16:00:00 2025

@author: Mir Imtiaz Mostafiz
@author: Jincheng Zhou
"""
import os
import sys
import logging
import argparse
import torch
import numpy as np
import random
import utils
import mnist as mnist
import NeuralNetwork as mynn
import minibatcher
import HamiltonianMonteCarlo as hmc
import PerturbedHamiltonianMonteCarlo as phmc

shallow_shape = None
deep_shape = None

# Set quick flushing for slurm output
sys.stdout.reconfigure(line_buffering=True, write_through=True)


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for MNIST')

    parser.add_argument('data_folder', metavar='DATA_FOLDER',
                        help='the folder that contains all the input data')
    parser.add_argument('-q', '--n_training_examples', type=int, default=-1,
                        help='max number of examples for training (-1 means all). (DEFAULT: -1)')
    parser.add_argument('-x', '--n_test_examples', type=int, default=-1,
                        help='max number of examples for test (-1 means all). (DEFAULT: -1)')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help='gpu id to use. -1 means cpu (DEFAULT: 0)')
    parser.add_argument('-s', '--depth', choices=['deep', 'shallow'], default='shallow',
                        help='choose the network Shape (DEFAULT: shallow)')
    parser.add_argument('-m', '--batch_size', type=int, default=10000,
                        help='minibatch_size. -1 means all. (DEFAULT: 10000)')
    parser.add_argument('-l', '--lr', type=float, default=1e-4,
                        help='learning rate for gradient descent. (DEFAULT: 1e-4)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='max number of epochs for training each model. (DEFAULT: 100)')

    parser.add_argument('--num_samples', default=10,
                        help='number of samples generated from HMC (DEFAULT:10)')
    parser.add_argument('--std_dev', default=1,
                        help='standard deviation for HMC sampling (DEFAULT:1)')
    parser.add_argument('--delta', type=float, default=0.0005,
                        help='delta value in HMC algorithm. (DEFAULT: 0.0005)')
    parser.add_argument('--num_leapfrog', type=int, default=8,
                        help='number of leapfrog steps in hmc. (DEFAULT: 8)')
    parser.add_argument('--p_num_samples', default=10,
                        help='number of samples generated from perturbed HMC (DEFAULT:10)')
    parser.add_argument('--p_num_leapfrog', type=int, default=12,
                        help='number of leapfrog steps in perturbed hmc. (DEFAULT: 12)')
    parser.add_argument('--num_bin', default=10,
                        help='number of bins in reliability analysis (DEFAULT:10')

    parser.add_argument('--loaded', type=int, default=0,
                        help='If the model is saved, pass 1 to load model from file;pass 0 for learning the model \
                        and then save it')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')

    args = parser.parse_args(argv)

    if args.loaded == 0:
        utils.loaded = False
    else:
        utils.loaded = True

    if args.gpu_id == -1:
        utils.device = torch.device("cpu")
        utils.thrng = torch.Generator(device="cpu")
    else:
        utils.device = torch.device("cuda:0")
        utils.thrng = torch.Generator(device="cuda")


    utils.thrng = torch.Generator("cpu")

    return args


def load_variables_default():
    utils.data_folder = "data"
    utils.n_training_examples = 10000
    utils.n_test_examples = 1000
    utils.gpu_id = 0
    utils.depth = "shallow"  # "deep"
    utils.batch_size = 10000
    utils.device = torch.device("cpu")
    utils.lr = 1e-4
    utils.epochs = 100
    utils.thrng = torch.Generator("cpu")
    utils.num_samples = 10
    utils.std_dev = 1
    utils.delta = 0.0005
    utils.num_leapfrog = 8
    utils.is_binary_class = True
    utils.loaded = False
    utils.p_num_samples = 10
    utils.p_num_leapfrog = 12
    utils.num_bin = 10


def set_variables(args):
    utils.data_folder = args.data_folder
    utils.n_training_examples = args.n_training_examples
    utils.n_test_examples = args.n_test_examples
    utils.gpu_id = args.gpu_id
    utils.depth = args.depth
    utils.batch_size = args.batch_size
    utils.lr = args.lr
    utils.epochs = args.epochs
    utils.num_samples = args.num_samples
    utils.std_dev = args.std_dev
    utils.delta = args.delta
    utils.num_leapfrog = args.num_leapfrog
    utils.is_binary_class = True
    utils.loaded = args.loaded
    utils.p_num_samples = args.p_num_samples
    utils.p_num_leapfrog = args.p_num_leapfrog
    utils.num_bin = args.num_bin


def get_model_shape():
    """
    decides the shape of the network, if "shallow", 1 hidden layer. else if "deep", 2 hidden layers

    Returns
    -------
    shape : List of integers of size == layers
        list(input nodes, hidden layer nodes, output classes).

    """
    global shallow_shape, deep_shape

    if utils.depth == "shallow":
        shape = shallow_shape
    elif utils.depth == "deep":
        shape = deep_shape
    else:
        shape = shallow_shape

    return shape


def data_pre_process(X_train, y_train, X_test, y_test):
    """
    Process train and test data before passing to the training pipeline

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
    # reshape the images into one dimension
    X_train, y_train_1hot, X_test, y_test_1hot = utils.reshape_data(X_train, y_train, X_test, y_test)

    # to torch tensor
    X_train, y_train, y_train_1hot, X_test, y_test, y_test_1hot = utils.torchify_tensors(X_train, y_train, y_train_1hot,
                                                                                         X_test, y_test, y_test_1hot)

    # to torch variable and cuda tensor
    X_train, y_train, y_train_1hot, X_test, y_test, y_test_1hot = utils.convert_tensors_variable_gpu(X_train, y_train,
                                                                                                     y_train_1hot,
                                                                                                     X_test, y_test,
                                                                                                     y_test_1hot,
                                                                                                     utils.gpu_id)

    return X_train, y_train, y_train_1hot, X_test, y_test, y_test_1hot


def main():
    # fixing all the random seeds

    seed_value = 12347
    utils.fix_seeds(seed_value)

    """
    Step 1: Data Processing
    """

    # Downloading MNIST data

    X_train, y_train, X_test, y_test = utils.load_mnist(utils.data_folder, utils.n_training_examples,
                                                        utils.n_test_examples, utils.is_binary_class)

    # Preprocess data

    X_train, y_train, y_train_1hot, X_test, y_test, y_test_1hot = data_pre_process(X_train, y_train, X_test, y_test)

    # number of training examples
    n_examples_train = X_train.data.shape[0]



    if os.path.exists('./plots/') == False:
        os.mkdir('./plots/')

    if os.path.exists('./models/') == False:
        os.mkdir('./models/')

    """
    Step 2: Training Pipeline
    """

    """
    Step 2a: Training Prior weights
    """
    # model shapes
    global shallow_shape, deep_shape
    shallow_shape = [X_train.shape[1], 200, mnist.N_CLASSES]
    deep_shape = [X_train.shape[1], 300, 100, mnist.N_CLASSES]

    shape = get_model_shape()

    mymodel = mynn.NeuralNetwork(shape, learning_rate=utils.lr)
    print(mymodel)

    if utils.gpu_id != -1:
        utils.device = torch.device("cuda:0")

    mymodel.to(utils.device)

    # Minibatch module loading

    if utils.loaded == False:
        """
        learn model and save it in disk
        """

        batcher = minibatcher.MiniBatcher(batch_size=utils.batch_size, n_examples=n_examples_train, shuffle=True)

        losses = []
        train_accs = []
        test_accs = []

        for epoch in range(utils.epochs):
            logging.info("---------- EPOCH {} ----------".format(epoch))

            for idxs in batcher.get_one_batch():
                # print(idxs)
                idxs = idxs.to(utils.device)
                loss = mymodel.train_one_epoch(X_train[idxs], y_train[idxs], y_train_1hot[idxs])
                logging.info("MLE loss = {}".format(loss))
                # monitor training and testing accuracy
                y_train_pred = mymodel.predict(X_train)
                y_test_pred = mymodel.predict(X_test)
                train_acc = utils.accuracy(y_train, y_train_pred)
                test_acc = utils.accuracy(y_test, y_test_pred)
                logging.info("MLE Accuracy(train) = {}".format(train_acc))
                logging.info("MLE Accuracy(test) = {}".format(test_acc))

            # collect results for plotting for each epoch
            loss = mymodel.loss(X_train, y_train_pred, y_train_1hot)
            losses.append(loss)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

        torch.save(mymodel.state_dict(), './models/' + utils.depth + 'model.pth')

        """
        Plot loss-accuracy graphs with filename appended to the x-axis label
        """
        utils.plot_graph([i + 1 for i in range(utils.epochs)],
                        train_accs,
                        "Epochs (File: " + utils.depth + "_mle_train_accuracy.jpg)",
                        "Accuracy",
                        "Train Epoch vs Accuracy",
                        './plots/' + utils.depth + "_mle_train_accuracy.jpg")
        utils.plot_graph([i + 1 for i in range(utils.epochs)],
                        test_accs,
                        "Epochs (File: " + utils.depth + "_mle_test_accuracy.jpg)",
                        "Accuracy",
                        "Test Epoch vs Accuracy",
                        './plots/' + utils.depth + "_mle_test_accuracy.jpg")
        utils.plot_graph([i + 1 for i in range(utils.epochs)],
                        losses,
                        "Epochs (File: " + utils.depth + "_mle_train_loss.jpg)",
                        "Loss",
                        "Train Epoch vs Loss",
                        './plots/' + utils.depth + "_mle_train_loss.jpg")


    elif utils.loaded == True:
        """
        Model is saved just need to be loaded from disc
        """
        mymodel.load_state_dict(torch.load('./models/' + utils.depth + 'model.pth'))

    """
    Step 2b: Initialize and run HMC Sampler
    """

    param_sampler = hmc.HamiltonianMonteCarloSampler(mymodel, utils.thrng, utils.device)

    samples = param_sampler.sample(utils.num_samples, utils.std_dev,
                                utils.delta, utils.num_leapfrog, X_train, y_train, y_train_1hot)

    """
    Step 2c: Run the sampled models for predictions
    """

    bayessian_y_train_prob = torch.zeros((X_train.shape[0], mnist.N_CLASSES))
    bayessian_y_test_prob = torch.zeros((X_test.shape[0], mnist.N_CLASSES))

    for (i, sampled_parameters) in enumerate(samples):

        new_bayesian_model = mynn.NeuralNetwork(shape, learning_rate=utils.lr)
        new_bayesian_model.to(utils.device)

        for (param, init) in zip(new_bayesian_model.parameters(), sampled_parameters):
            param.data.copy_(init)

        train_prob = new_bayesian_model.get_prob(X_train).cpu()
        test_prob = new_bayesian_model.get_prob(X_test).cpu()
        bayessian_y_train_prob.add_(train_prob)
        bayessian_y_test_prob.add_(test_prob)

    bayessian_y_train_prob /= len(samples)
    bayessian_y_test_prob /= len(samples)

    bayesian_y_train_pred = torch.max(bayessian_y_train_prob, 1)[1].to(utils.device)
    bayesian_y_test_pred = torch.max(bayessian_y_test_prob, 1)[1].to(utils.device)

    bayesian_train_acc = utils.accuracy(y_train, bayesian_y_train_pred)
    bayesian_test_acc = utils.accuracy(y_test, bayesian_y_test_pred)
    bayesian_loss = new_bayesian_model.loss(X_train, bayesian_y_train_pred, y_train_1hot)

    logging.info("Bayessian loss = {}".format(bayesian_loss))
    logging.info("Bayessian Accuracy(train) = {}".format(bayesian_train_acc))
    logging.info("Bayessian Accuracy(test) = {}".format(bayesian_test_acc))

    """
    Step 2d: Initialize and run Perturbed HMC Sampler
    """

    perturbed_param_sampler = phmc.PerturbedHamiltonianMonteCarloSampler(mymodel, utils.thrng, utils.device)
    perturbed_samples = perturbed_param_sampler.sample(utils.p_num_samples, utils.std_dev,
                                                    utils.delta, utils.p_num_leapfrog, X_train, y_train,
                                                    y_train_1hot)

    """
    Step 2e: Run the sampled models from perturbed HMC for predictions
    """

    perturbed_bayessian_y_train_prob = torch.zeros((X_train.shape[0], mnist.N_CLASSES))
    perturbed_bayessian_y_test_prob = torch.zeros((X_test.shape[0], mnist.N_CLASSES))

    for (i, perturbed_sampled_parameters) in enumerate(perturbed_samples):

        perturbed_new_bayesian_model = mynn.NeuralNetwork(shape, learning_rate=utils.lr)
        perturbed_new_bayesian_model.to(utils.device)

        for (param, init) in zip(perturbed_new_bayesian_model.parameters(), perturbed_sampled_parameters):
            param.data.copy_(init)

        perturbed_train_prob = perturbed_new_bayesian_model.get_prob(X_train).cpu()
        perturbed_test_prob = perturbed_new_bayesian_model.get_prob(X_test).cpu()
        perturbed_bayessian_y_train_prob.add_(perturbed_train_prob)
        perturbed_bayessian_y_test_prob.add_(perturbed_test_prob)

    perturbed_bayessian_y_train_prob /= len(perturbed_samples)
    perturbed_bayessian_y_test_prob /= len(perturbed_samples)

    perturbed_bayesian_y_train_pred = torch.max(perturbed_bayessian_y_train_prob, 1)[1].to(utils.device)
    perturbed_bayesian_y_test_pred = torch.max(perturbed_bayessian_y_test_prob, 1)[1].to(utils.device)

    perturbed_bayesian_train_acc = utils.accuracy(y_train, perturbed_bayesian_y_train_pred)
    perturbed_bayesian_test_acc = utils.accuracy(y_test, perturbed_bayesian_y_test_pred)
    perturbed_bayesian_loss = perturbed_new_bayesian_model.loss(X_train, perturbed_bayesian_y_train_pred, y_train_1hot)

    logging.info("Perturbed_Bayessian loss = {}".format(perturbed_bayesian_loss))
    logging.info("Perturbed_Bayessian Accuracy(train) = {}".format(perturbed_bayesian_train_acc))
    logging.info("Perturbed_Bayessian Accuracy(test) = {}".format(perturbed_bayesian_test_acc))

    """
    Step 3: Draw ROC Curves
    """

    y_train_prob = mymodel.get_prob(X_train)
    y_test_prob = mymodel.get_prob(X_test)

    _, _, _, auc_mle_train = utils.plot_roc_auc_curve(y_train, y_train_prob,
                                                    './plots/' + utils.depth + "_mle_train_auc.jpg")
    _, _, _, auc_mle_test = utils.plot_roc_auc_curve(y_test, y_test_prob,
                                                    './plots/' + utils.depth + "_mle_test_auc.jpg")

    _, _, _, auc_bayessian_train = utils.plot_roc_auc_curve(y_train, bayessian_y_train_prob,
                                                            './plots/' + utils.depth + "_bayessian_train_auc.jpg")
    _, _, _, auc_bayessian_test = utils.plot_roc_auc_curve(y_test, bayessian_y_test_prob,
                                                        './plots/' + utils.depth + "_bayessian_test_auc.jpg")

    _, _, _, auc_perturbed_bayessian_train = utils.plot_roc_auc_curve(y_train, perturbed_bayessian_y_train_prob,
                                                                    './plots/' + utils.depth + "_perturbed_bayessian_train_auc.jpg")
    _, _, _, auc_perturbed_bayessian_test = utils.plot_roc_auc_curve(y_test, perturbed_bayessian_y_test_prob,
                                                                    './plots/' + utils.depth + "_perturbed_bayessian_test_auc.jpg")

    """
    Step 4: Draw Reliability Diagrams
    """

    logging.info("ECE for MLE in training:")
    utils.analyse_calibration(y_train, y_train_prob, utils.num_bins,
                            './plots/' + utils.depth + "_mle_train_reliablity.jpg")
    logging.info("ECE for HMC in training:")
    utils.analyse_calibration(y_train, bayessian_y_train_prob, utils.num_bins,
                            './plots/' + utils.depth + "_bayessian_train_reliablity.jpg")
    logging.info("ECE for pertubated HMC in training:")
    utils.analyse_calibration(y_train, perturbed_bayessian_y_train_prob, utils.num_bins,
                            './plots/' + utils.depth + "_perturbed_bayessian_train_reliablity.jpg")

    logging.info("ECE for MLE in test:")
    utils.analyse_calibration(y_test, y_test_prob, utils.num_bins,
                            './plots/' + utils.depth + "_mle_test_reliablity.jpg")
    logging.info("ECE for HMC in test:")
    utils.analyse_calibration(y_test, bayessian_y_test_prob, utils.num_bins,
                            './plots/' + utils.depth + "_bayessian_test_reliablity.jpg")
    logging.info("ECE for pertubated HMC in test:")
    utils.analyse_calibration(y_test, perturbed_bayessian_y_test_prob, utils.num_bins,
                            './plots/' + utils.depth + "_perturbed_bayessian_test_reliablity.jpg")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        load_variables_default()
    else:
        args = utils.bin_config(get_arguments)
        set_variables(args)

    main()
