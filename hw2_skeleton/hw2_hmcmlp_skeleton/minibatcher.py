# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 00:39:02 2022

@author: Mir Imtiaz Mostafiz
"""
import numpy as np
import torch
import logging


# TODO: make full torch implementation
class MiniBatcher(object):
    def __init__(self, batch_size, n_examples, shuffle=True):
        assert batch_size <= n_examples, "Error: batch_size is larger than n_examples"
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.shuffle = shuffle
        logging.info("batch_size={}, n_examples={}".format(batch_size, n_examples))

        if shuffle:
            self.idxs = torch.randperm(self.n_examples)
        else:
            self.idxs = torch.arange(self.n_examples)
        self.current_start = 0

    def get_one_batch(self):
        
        self.current_start = 0
        while self.current_start < self.n_examples:
            batch_idxs = self.idxs[self.current_start:self.current_start+self.batch_size]
            self.current_start += self.batch_size
            yield torch.LongTensor(batch_idxs)
