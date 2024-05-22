import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class User2:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, beta = 0 , L_k = 0, local_epochs = 0):
        # from fedprox
        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.L_k = L_k
        self.local_epochs = local_epochs
        
        
    def set_commonPCA(self, commonPCA):
        #for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
        self.localZ = commonPCA
        self.localLamda = self.localLamda + self.ro * (self.localPCA - self.localZ)

    def test(self):
        self.model.eval()
        loss_test = 0
        loss_test = self.loss()
        return loss_test, self.test_samples

    def train_error_and_loss(self):
        self.model.eval()
        loss_train = self.loss()
        return loss_train , self.train_samples
        
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
