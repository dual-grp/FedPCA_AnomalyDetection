import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from torchmetrics import ConfusionMatrix

import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Implementation for FedAvg clients

class UserAvg(User):
    def __init__(self, device, numeric_id, train_data, test_data, test_abnormal_data,
                 model, batch_size, learning_rate, local_epochs):
        super().__init__(device, numeric_id, train_data, test_data,
                         model[0], batch_size, learning_rate, 0, 0, local_epochs)

        if (model[1] == "ae" or model[1] == "ae2"):
            self.loss = torch.nn.MSELoss()
        elif (model[1] == "bigan"):
            self.loss = torch.nn.BCELoss()
        else:
            self.loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.learning_rate)
        
        self.confmat = ConfusionMatrix(task='binary', num_classes=2).to(device)
        #print(self.local_epochs)
        #print(self.batch_size)

        #test_normal = [(X, y) for (X, y) in test_data if y == 0]
        #test_abnormal = [(X, y) for (X, y) in test_data if y == 1]

        #print("Normal Test:", len(test_normal))
        #print("Abnormal Test:", len(test_abnormal))

        #print(len(test_normal))
        #print(len(test_abnormal))

        self.test_normal_loader = DataLoader(test_data, len(test_data), shuffle=False)
        self.test_abnormal_loader = DataLoader(test_abnormal_data, len(test_abnormal_data),shuffle=False)
        

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        losses = []
        #print(self.local_epochs)
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for X,y in self.trainloader:
                #print("here")
                X, y = X.to(self.device), y.to(self.device)#self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, X)
                losses.append(loss)
                #print(loss)
                loss.backward()
                self.optimizer.step()
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return losses

    def confusion_matrix_torch(self, y_true, y_pred):
        """
        Compute confusion matrix and return TN, FP, FN, TP
        """
        # Ensure the tensors are on the same device
        device = y_true.device
        #y_pred = y_pred.to(device)
        # Get the labels: predicted and true ones
        #y_pred = torch.round(torch.sigmoid(y_pred))
        #y_true = y_true.long()
        # Initialize confusion matrix
        conf_matrix = torch.zeros(2, 2).to(device)

        with torch.no_grad():
            for t, p in zip(y_true.view(-1), y_pred.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

        tp = conf_matrix[0, 0].item()
        fn = conf_matrix[0, 1].item()
        fp = conf_matrix[1, 0].item()
        tn = conf_matrix[1, 1].item()

        return tn, fp, fn, tp

    def find_optimal_threshold(self, A, B):
        # Combine and sort both lists
        combined = sorted(A + B)

        max_count = 0
        optimal_threshold = None

        # Iterate through all possible threshold values in the combined list
        for t in combined:
            # Count the number of elements in A that are less than t
            count_A = sum(i < t for i in A)
            # Count the number of elements in B that are greater than or equal to t
            count_B = sum(i >= t for i in B)

            # If the sum of count_A and count_B is greater than the maximum count found so far
            if count_A + count_B > max_count:
                max_count = count_A + count_B
                optimal_threshold = t

        return optimal_threshold, max_count

    def find_optimal_threshold_2(self, A, B):
        A = sorted(A)
        B = sorted(B)

        n, m = len(A), len(B)
        i, j = 0, 0

        max_count = 0
        optimal_threshold = None

        while i < n and j < m:
            if A[i] < B[j]:
                count_A = i + 1
                count_B = m - j
                i += 1
            else:  # when A[i] >= B[j]
                count_A = i
                count_B = m - j + 1
                j += 1

            if count_A + count_B > max_count:
                max_count = count_A + count_B
                optimal_threshold = A[i - 1] if i > 0 else B[j - 1]

        # If there are remaining elements in either list, we handle them here
        while i < n:
            count_A = i + 1
            count_B = 0
            if count_A + count_B > max_count:
                max_count = count_A + count_B
                optimal_threshold = A[i]
            i += 1

        while j < m:
            count_A = 0
            count_B = m - j + 1
            if count_A + count_B > max_count:
                max_count = count_A + count_B
                optimal_threshold = B[j]
            j += 1

        return optimal_threshold, max_count
    
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
                #print("1")
                x, y = x.to(self.device), y.to(self.device)
                true_labels.append(y)
                output = self.model(x)
                #loss = self.loss(output, y)
                #test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                error = ((output - x)**2).mean(dim=1)
                normal_loss += self.loss(output, x)
                reconstruction_normal_errors.append(error)
                pred = (error > threshold).type(torch.int)  # 1 if anomaly, 0 otherwise
                predictions.extend(pred.tolist())
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)

            for x, y in self.test_abnormal_loader:
                #print("2")
                x, y = x.to(self.device), y.to(self.device)
                true_labels[0] = torch.cat((true_labels[0], y))
                output = self.model(x)
                #loss = self.loss(output, y)
                #test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                error = ((output - x)**2).mean(dim=1)
                abnormal_loss += self.loss(output, x)
                reconstruction_abnormal_errors.append(error)
                pred = (error > threshold).type(torch.int)  # 1 if anomaly, 0 otherwise
                predictions.extend(pred.tolist())
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)
        
        #print(abnormal_loss)
       
        res_errors = reconstruction_normal_errors[0].data.tolist()
        res_abnormal_errors = reconstruction_abnormal_errors[0].data.tolist()
        res_errors = res_errors + res_abnormal_errors
        #print(len(res_errors))
        #print(len(res_abnormal_errors))
        #print(len(res_errors + res_abnormal_errors))
        """
        pd_normal_errors = pd.Series(res_errors)
        pd_abnormal_erros = pd.Series(res_abnormal_errors)

        # pd_normal_errors.min(), pd_normal_errors.max()
        
        print(pd_normal_errors.min())
        print(pd_normal_errors.max())
        print(pd_normal_errors.value_counts(bins=np.round(np.linspace(0, 0.0025, 21), decimals=10), sort=False))

        print(pd_abnormal_erros.min())
        print(pd_abnormal_erros.max())       
        print(pd_abnormal_erros.value_counts(bins=np.round(np.linspace(0, 0.0025, 21), decimals=10), sort=False))        
        """
        #opt_normal_threshold = np.percentile(res_errors, 95)
        #opt_abormal_threshold = np.percentile(res_abnormal_errors, 15)
        #opt_threshold, max_count = self.find_optimal_threshold(res_errors, res_abnormal_errors)
        #print("Optimal Threshold : ", opt_threshold)
        #print("Max count : ", max_count)

        #print("Test normal threshold : ", opt_normal_threshold)
        #print("Test abnormal threshold : ", opt_abormal_threshold)
        #print(pd.Series(A).hist(bins=2))
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
        # Compute metrics
        #accuracy = accuracy_score(true_labels, predictions)
        #precision = precision_score(true_labels, predictions)
        #recall = recall_score(true_labels, predictions)
        #f1 = f1_score(true_labels, predictions)
        #fnr = fn / (fn + tp)
        #return test_acc, y.shape[0], test_acc / y.shape[0]
        return tn, fp, fn, tp, \
            normal_loss.data.tolist(), abnormal_loss.data.tolist(),  \
                predictions.data.tolist(), true_labels[0].data.tolist(), self.test_samples, \
                res_errors

    def c_train_error_and_loss(self, threshold):
        self.model.eval()
        train_acc = 0
        loss = 0
        reconstruction_errors = []
        predictions = []
        true_labels = []
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                true_labels.append(y)
                output = self.model(x)
                #train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                error = ((output - x)**2).mean(dim=1)
                reconstruction_errors.append(error)
                loss += self.loss(output, x)
                pred = (error > threshold).type(torch.int)  # 1 if anomaly, 0 otherwise
                predictions.extend(pred.tolist())
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)

        ##opt_threshold = np.percentile(res_errors, 95)
        
        #print("Train Normal Threshold : ", opt_threshold)

        #counts = pd.cut(df.score, np.round(np.arange(-1,1.1,0.1),2))    
        #counts.value_counts().sort_index()
        #print(pd.Series(reconstruction_errors).value_counts().sort_index())
        #print(pd.Series(predictions).value_counts())
        predictions = torch.Tensor(predictions).to(self.device)
        #print(predictions)

         # Compute confusion matrix
        #tn, fp, fn, tp = self.confusion_matrix_torch(true_labels[0], predictions)

        #confmat = ConfusionMatrix(num_classes=2)
        conf_matrix = self.confmat(predictions, true_labels[0])
        #print(conf_matrix)
        tn = conf_matrix[0, 0].item()
        fp = conf_matrix[0, 1].item()
        fn = conf_matrix[1, 0].item()
        tp = conf_matrix[1, 1].item()
        
        return tn, fp, fn, tp, loss.data.tolist(), self.train_samples