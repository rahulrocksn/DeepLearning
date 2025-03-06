# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 00:10:23 2022

@author: Mir Imtiaz Mostafiz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG_NN = False

class NeuralNetwork(nn.Module):
    """
    Pytorch implementation of a Image classifier Neural Network
    """
    def __init__(self, shape, learning_rate, gpu_id = -1):
        
        super(NeuralNetwork, self).__init__()
        self.shape = shape
        moduleLists = []
        
        for i in range(len(shape)-1):
            
            moduleLists.append(nn.Linear(self.shape[i], self.shape[i+1]))
            #moduleLists.append(nn.ReLU())
            
        #moduleLists.append(nn.LogSoftmax(dim=1))
        
        self.layers = nn.ModuleList(moduleLists)
        
        self.loss_fn = nn.NLLLoss()
        
        self.gpu_id = gpu_id
        
        self.learning_rate = learning_rate
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        
    def __repr__(self):
        """
        String representation

        Returns
        -------
        ret : String
            string representation of the model.

        """
        
        ret = "The model has:\n"
        
        for layer in self.layers:
            
            ret += str(layer) + "\n"
            
        ret += "\n\n"
        
        return ret
    
    def forward(self, x):
        """
        Forward pass of the data into the model

        Parameters
        ----------
        x : torch.tensor, shape (num_examples, features)
            training data.

        Returns
        -------
        x : torch.tensor, shape (num_examples, )
            forward pass value (log probability).

        """
        
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            if DEBUG_NN:
                print(x.shape)
        
        x = self.layers[-1](x)        
        x = F.log_softmax(x,dim=1)
        return x
    
    def train_one_epoch(self, X, y, y_1hot):
        """
        Training of (batched) data for one epoch 

        Parameters
        ----------
        X : torch.tensor, shape (num_examples, features)
            training data
        y : torch.tensor, shape (num_examples, )
            training target
        y_1hot : torch.tensor, shape (num_examples, classes)
            training target one hot encoding

        Returns
        -------
        loss: integer
            training loss.

        """
        
        X_train = X
        y_train = y

        # forward
        y_pred = self.forward(X_train)
        #logging.info("input to loss = {}".format(y_pred))
        loss = self.loss_fn(y_pred, y_train.squeeze())

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()
    
    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        
        y_pred = self.forward(X)
        loss = self.loss_fn(y_pred, y.squeeze())
    
        return loss.cpu().item()

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        
            
        y_pred = self.forward(X)
        
        return torch.max(y_pred, 1)[1]

    def get_prob(self, X):
        
        y_pred = self.forward(X)
        
        return torch.exp(y_pred)
    
    def energy(self, X, y, y_1hot):
        """Compute HMC potential energy: same as negative log-likelihood theoretically

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Energy for the given input
        """

        y_pred = self.forward(X)
        energy = self.loss_fn(y_pred, y.squeeze())
    
        
        return energy
        