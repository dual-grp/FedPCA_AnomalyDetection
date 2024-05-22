import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from torchmetrics import ConfusionMatrix
torch.manual_seed(25)

import os
import json
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Implementation for FedAvg clients

class UserBiGAN(User):
    def __init__(self, device, numeric_id, train_data, test_data, test_abnormal_data,
                 model, latent_dim, batch_size, learning_rate, local_epochs):
        super().__init__(device, numeric_id, train_data, test_data,
                         model[0], batch_size, learning_rate, 0, 0, local_epochs)

        #if (model[1] == "ae"):
        #    self.loss = torch.nn.MSELoss()
        #elif (model[1] == "bigan"):
        #    self.loss = torch.nn.BCELoss()
        #else:
        #    self.loss = nn.NLLLoss()
        self.loss = torch.nn.BCELoss()
        self.latent_dim = latent_dim
        self.confmat = ConfusionMatrix(task='binary', num_classes=2).to(device)
        #print(self.local_epochs)
        #print(self.batch_size)

        #test_normal = [(X, y) for (X, y) in test_data if y == 0]
        #test_abnormal = [(X, y) for (X, y) in test_data if y == 1]

        #print("Normal Test:", len(test_normal))
        #print("Abnormal Test:", len(test_abnormal))

        self.test_normal_loader = DataLoader(test_data, len(test_data), shuffle=False)
        self.test_abnormal_loader = DataLoader(test_abnormal_data, len(test_abnormal_data),shuffle=False)
        if self.batch_size == 0:
            self.train_normal_loader = DataLoader(train_data, self.train_samples, shuffle=True)
        else:
            self.train_normal_loader = DataLoader(train_data, self.batch_size, shuffle=True, drop_last=True)
       
        self.optimizer_g = torch.optim.SGD(self.model.generator.parameters(),
                                         lr=self.learning_rate)
        self.optimizer_e = torch.optim.SGD(self.model.encoder.parameters(),
                                         lr=self.learning_rate)
        self.optimizer_d = torch.optim.SGD(self.model.discriminator.parameters(),
                                         lr=self.learning_rate)               
        #print(self.local_epochs)
        
        #print(self.batch_size)
        self.real_labels = torch.ones(self.batch_size, 1).to(device)
        self.fake_labels = torch.zeros(self.batch_size, 1).to(device)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        #print(self.local_epochs)
        self.model.generator.train()
        self.model.discriminator.train()
        self.model.encoder.train()

        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for X,y in self.train_normal_loader:
                #print("here")
                X, y = X.to(self.device), y.to(self.device)
                #self.get_next_train_batch()
                #batch_size = X.shape[0]
                # Forward
                z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
                real, fake = self.model(X, z)

                # Train Discriminator
                self.optimizer_d.zero_grad()
                loss_real = self.loss(real, self.real_labels)
                loss_fake = self.loss(fake, self.fake_labels)
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward(retain_graph=True)
                self.optimizer_d.step()

                #  Generator
                self.optimizer_g.zero_grad()
                self.optimizer_e.zero_grad()
                #z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
                #real, fake = self.model(X, z)
                loss_real = self.loss(real, self.fake_labels)
                loss_fake = self.loss(fake, self.real_labels)
                loss_G = (loss_real + loss_fake) / 2
                #loss_G = self.loss(fake, self.real_labels)
                loss_G.backward(retain_graph=True)
                self.optimizer_g.step()
                self.optimizer_e.step()

        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS
    
    
    def c_test(self, threshold):
        self.model.eval()
        test_acc = 0
        normal_loss = 0
        abnormal_loss = 0
        reconstruction_normal_errors = []
        reconstruction_abnormal_errors = []
        predictions = []
        true_labels = []
        with torch.no_grad():
            for x, y in self.test_normal_loader:
                x, y = x.to(self.device), y.to(self.device)
                true_labels.append(y)
                z = self.model.encoder(x)
                output = self.model.generator(z)
                #loss = self.loss(output, y)
                #test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #error = ((output - x)**2).mean(dim=1)
                error = torch.norm(x - output, dim=1)
                normal_loss += torch.mean(error)
                reconstruction_normal_errors.append(error)
                pred = (error > threshold).type(torch.int)  # 1 if anomaly, 0 otherwise
                predictions.extend(pred.tolist())
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)

            for x, y in self.test_abnormal_loader:
                x, y = x.to(self.device), y.to(self.device)
                true_labels[0] = torch.cat((true_labels[0], y))
                z = self.model.encoder(x)
                output = self.model.generator(z)
                #output = self.model(x)
                #loss = self.loss(output, y)
                #test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #error = ((output - x)**2).mean(dim=1)
                error = torch.norm(x - output, dim=1)
                abnormal_loss += torch.mean(error)
                reconstruction_abnormal_errors.append(error)
                pred = (error  > threshold).type(torch.int)  # 1 if anomaly, 0 otherwise
                predictions.extend(pred.tolist())
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)

        
        res_normal_errors = reconstruction_normal_errors[0].data.tolist()
        res_abnormal_errors = reconstruction_abnormal_errors[0].data.tolist()
        res_errors = res_normal_errors + res_abnormal_errors
        #print(len(res_errors))
        #print(len(res_abnormal_errors))
        #print(len(res_errors + res_abnormal_errors))
        """
        pd_normal_errors = pd.Series(res_errors)
        pd_abnormal_erros = pd.Series(res_abnormal_errors)

        # pd_normal_errors.min(), pd_normal_errors.max()
        
        print(pd_normal_errors.min())
        print(pd_normal_errors.max())
        print(pd_normal_errors.value_counts(bins=np.round(np.linspace(pd_normal_errors.min(), 
                                                                      pd_normal_errors.max(), 21),
                                                          decimals=10), sort=False))

        print(pd_abnormal_erros.min())
        print(pd_abnormal_erros.max())       
        print(pd_abnormal_erros.value_counts(bins=np.round(np.linspace(pd_abnormal_erros.min(),
                                                                       pd_abnormal_erros.max(), 21),
                                                           decimals=10), sort=False))        
        
        """
        #print(true_labels)
        #print(normal_loss.data.tolist())
        #print(abnormal_loss.data.tolist())
        #print(pd.Series(predictions).value_counts())
        predictions = torch.Tensor(predictions).to(self.device)
        #print(true_labels[0])
        #print(predictions)

        # Compute confusion matrix
        #tn, fp, fn, tp = self.confusion_matrix_torch(true_labels[0], predictions)
        conf_matrix = self.confmat(predictions, true_labels[0])
        #print(conf_matrix)
        tn = conf_matrix[0, 0].item()
        fp = conf_matrix[0, 1].item()
        fn = conf_matrix[1, 0].item()
        tp = conf_matrix[1, 1].item()

        return tn, fp, fn, tp, \
                normal_loss.data.tolist(), abnormal_loss.data.tolist(), \
                predictions.data.tolist(), true_labels[0].data.tolist(), self.test_samples, \
                res_errors

    def c_train_error_and_loss(self, threshold):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.mean_loss = 0
        reconstruction_errors = []
        predictions = []
        true_labels = []
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                true_labels.append(y)
                z = self.model.encoder(x)
                output = self.model.generator(z)
                #output = self.model(x)
                #train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #error = ((output - x)**2).mean(dim=1)
                error = torch.norm(x - output, dim=1)
                loss += torch.mean(error)
                reconstruction_errors.append(error)

                pred = (error > threshold).type(torch.int)  # 1 if anomaly, 0 otherwise
                predictions.extend(pred.tolist())
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
        
        self.mean_loss = loss
        #print(pd.Series(predictions).value_counts())
        predictions = torch.Tensor(predictions).to(self.device)
        #print(predictions)

        conf_matrix = self.confmat(predictions, true_labels[0])
        #print(conf_matrix)
        tn = conf_matrix[0, 0].item()
        fp = conf_matrix[0, 1].item()
        fn = conf_matrix[1, 0].item()
        tp = conf_matrix[1, 1].item()
        
        return tn, fp, fn, tp, loss.data.tolist(), self.train_samples
    
    # def train(self, epochs):
    #     LOSS = 0
    #     self.model.train()
    #     for epoch in range(1, self.local_epochs + 1):
    #         self.model.train()
    #         X, y = self.get_next_train_batch()
    #         self.optimizer.zero_grad()
    #         output = self.model(X)
    #         loss = self.loss(output, y)
    #         loss.backward()
    #         self.optimizer.step()
    #         self.clone_model_paramenter(self.model.parameters(), self.local_model)
    #     return LOSS