import torch
import os
import json
from FLAlgorithms.users.userbase import User
import copy

'''Implementation for FedPCA clients''' 

class UserADMM2():
    def __init__(self, algorithm, device, id, train_data, commonPCA, learning_rate, ro, local_epochs, dim):
        self.localPCA   = copy.deepcopy(commonPCA) # local U
        self.localZ     = copy.deepcopy(commonPCA)
        self.localY     = copy.deepcopy(commonPCA)
        self.localT     = torch.matmul(self.localPCA.T, self.localPCA)
        self.ro = ro
        self.device = device
        self.id = id
        self.train_samples = len(train_data)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.dim = dim
        self.train_data = train_data.T # Since we learn data representation in a lower dimension, the data is convert from [Nxd] to [dxN]. Then, d is reduced to k by FedPG and FedPE
        self.algorithm = algorithm
        self.localPCA.requires_grad_(True)

    def set_commonPCA(self, commonPCA):
        # update local Y
        self.localZ = commonPCA.data.clone()
        self.localY = self.localY + self.ro * (self.localPCA - self.localZ)
        # update local T
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
        hU = torch.max(torch.zeros(temp.shape),temp)**2
        self.localT = self.localT + self.ro * hU

    def train_error_and_loss(self):
        residual = torch.matmul((torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)), self.train_data)
        loss_train = torch.norm(residual, p="fro") ** 2 / self.train_samples
        return loss_train , self.train_samples

    def hMax(self):
        temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
        return torch.max(torch.zeros(temp.shape),temp)

    def train(self, epochs):
        for i in range(self.local_epochs):
            '''Euclidean space'''
            if self.algorithm == "FedPE": 
                self.localPCA.requires_grad_(True)
                residual = torch.matmul(torch.eye(self.localPCA.shape[0])- torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
                temp = torch.matmul(self.localPCA.T, self.localPCA) - torch.eye(self.localPCA.shape[1])
                hU = torch.max(torch.zeros(temp.shape),temp)**2
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)** 2 + 0.5 * self.ro * torch.norm(hU) ** 2
                frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ)) + torch.sum(torch.inner(self.localT, hU))
                self.loss = 1/self.train_samples * torch.norm(residual, p="fro") ** 2 
                self.lossADMM = self.loss + 1/self.train_samples * (frobenius_inner + regularization)
                temp = self.localPCA.data.clone()
                # Solve the local problem
                if self.localPCA.grad is not None:
                    self.localPCA.grad.data.zero_()

                self.lossADMM.backward(retain_graph=True)
                # Update local pca
                temp  = temp - self.learning_rate * self.localPCA.grad
                self.localPCA = temp.data.clone()
                 
            else: 
                '''Grassmannian manifold'''
                self.localPCA.requires_grad_(True)
                residual = torch.matmul(torch.eye(self.localPCA.shape[0])- torch.matmul(self.localPCA, self.localPCA.T), self.train_data)
                frobenius_inner = torch.sum(torch.inner(self.localY, self.localPCA - self.localZ))
                regularization = 0.5 * self.ro * torch.norm(self.localPCA - self.localZ)** 2
                self.loss = 1/self.train_samples * torch.norm(residual, p="fro") ** 2
                self.lossADMM = self.loss + 1/self.train_samples * (frobenius_inner + regularization)
                temp = self.localPCA.data.clone()
                # solve local problem locally
                if self.localPCA.grad is not None:
                    self.localPCA.grad.data.zero_()
                self.lossADMM.backward(retain_graph=True)
                '''Moving on Grassmannian manifold'''
                # Projection on tangent space
                projection_matrix = torch.eye(self.localPCA.shape[0]) - torch.matmul(self.localPCA, self.localPCA.T)
                projection_gradient = torch.matmul(projection_matrix, self.localPCA.grad)
                temp = temp - self.learning_rate * projection_gradient
                # Exponential mapping to Grassmannian manifold by QR retraction
                q, r = torch.linalg.qr(temp)
                self.localPCA = q.data.clone()
        return  