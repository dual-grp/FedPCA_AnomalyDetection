import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy

class Server2:
    def __init__(self, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, times):
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.L_k = ro
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc = [], [], []
        self.times = times
        self.dim = dim

    def send_pca(self):
        assert (self.users is not None and len(self.users) > 0)
        # print("check Z", torch.matmul(self.commonPCAz.T,self.commonPCAz))
        # for user in self.users:
        for user in self.selected_users:
            # print("user_id", user.id)
            user.set_commonPCA(self.commonPCAz)
    
    def add_pca(self, user, ratio):
        # ADMM update
        # self.commonPCAz += ratio*(user.localPCA + 1/user.ro * user.localY)
        # simplified ADMM update
        # print("simplified ADMM update")
        self.commonPCAz += ratio*(user.localPCA)

    def aggregate_pca(self):
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
            # print("user_id", user.id)
        self.commonPCAz = torch.zeros(self.commonPCAz.shape)
        for user in self.selected_users:
            self.add_pca(user, user.train_samples / total_train)
    
    def select_users(self, round, fac_users):
        if(fac_users == 1):
            print("Distribute global model to all users")
            # for user in self.users:
            #     print("user_id", user.id)
            return self.users
        num_users = int(fac_users * len(self.users))
        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # Save loss, accurancy to h5 fiel
    def train_error_and_loss(self):
        num_samples = []
        losses = []
        for c in self.selected_users:
            cl, ns = c.train_error_and_loss() 
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]

        return ids, num_samples, losses

    def evaluate(self):
        stats_train = self.train_error_and_loss()
        # print(f"stats_train: {stats_train}")
        train_loss = sum(stats_train[2])/len(self.users)
        self.rs_train_loss.append(train_loss)
        if(self.experiment):
            self.experiment.log_metric("train_loss",train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Trainning Loss: ",train_loss)
        return train_loss
    
    def save_results(self):
        dir_path = "./results"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        alg = self.dataset[1] + "ADMM" + "_" + str(self.learning_rate)  + "_" + str(self.L_k) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs) 
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()

