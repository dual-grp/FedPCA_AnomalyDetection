import torch
import os
import torch.multiprocessing as mp

from FLAlgorithms.users.useravg import UserAvg
from FLAlgorithms.users.userbigan import UserBiGAN
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.models import *
from utils.model_utils import read_data, read_user_data
# from utils.model_utils import read_data, read_user_data
from utils.test_utils import unsw_nb15_test, nsl_kdd_test, iot23_test, ton_test
from utils.store_utils import print_log
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc

import matplotlib.pyplot as plt
# Implementation for FedAvg Server

class ServerAvg(Server):
    def __init__(self, algorithm, experiment, device, dataset, 
                 model, batch_size, learning_rate, num_glob_iters, 
                 local_epochs, dim, clients, fac_users, threshold, times):
        super().__init__(experiment, device, dataset, algorithm, 
                         model, batch_size, learning_rate, 0.0, 0.0, 
                         num_glob_iters,local_epochs, "sgd", clients, times)

        # Initialize data for all  users
        #print(fac_users)
        self.subusers = fac_users
        self.num_clients = clients # pre-define for 20 users
        self.dataset = dataset
        self.model_name = model
        self.latent_dim = dim
        self.out_dir = os.path.join(os.getcwd(), "baseline_results/")
        #self.timestamp = time.strftime('%Y-%m-%d %a %H:%M:%S')

        self.prefix = self.model_name + "_" + self.dataset + \
                        "_" + str(self.num_clients) + "_" + str(self.subusers) \
                        + "_" + str(self.learning_rate) + "_" + str(dim) \
                            + "_" + str(self.local_epochs) + "_"

        if self.dataset == "Unsw":
            dataX, dataX_test, dataX_abnormal = self.get_data_unsw_nb15()
            factor = len(dataX)/self.num_clients
            test_factor = len(dataX_test)/self.num_clients
            test_abnormal_factor = len(dataX_abnormal)/self.num_clients
            self.label_col = "attack_cat"
            self.label_name = "Normal"
            self.out_dir =  os.path.join(self.out_dir, "unsw/")
            #self.scaler = MinMaxScaler()
            self.scaler = StandardScaler()

        elif self.dataset == "Iot23":
            dataX = self.get_data_Iot23()
            factor = 29539/self.num_clients # 29539 is total number of data points in IoT23 dataset 
       
        elif self.dataset == "Ton":
            dataX, dataX_test, dataX_abnormal = self.get_data_Ton()
            factor = len(dataX)/self.num_clients
            test_factor = len(dataX_test)/self.num_clients
            test_abnormal_factor = len(dataX_abnormal)/self.num_clients
            self.label_col = "type"
            self.label_name = "normal"
            self.out_dir = os.path.join(self.out_dir, "ton/")
            self.scaler = MinMaxScaler()

        else:
            dataX = self.get_data_snl_kdd()
            factor = 67340/self.num_clients # 67340 is total number of data points in NSL KDD dataset

        #print(dataX.columns)
        self.scaler.fit(dataX[dataX.columns.difference([self.label_col])])

        dataX[dataX.columns.difference([self.label_col])] = self.scaler.transform(dataX[dataX.columns.difference([self.label_col])])
        dataX_test[dataX_test.columns.difference([self.label_col])] = self.scaler.transform(dataX_test[dataX_test.columns.difference([self.label_col])])
        dataX_abnormal[dataX_abnormal.columns.difference([self.label_col])] = self.scaler.transform(dataX_abnormal[dataX_abnormal.columns.difference([self.label_col])])
        #scaler.fit(df[df.columns.difference(['attack_cat', 'label'])])
  
        #df[df.columns.difference(['attack_cat', 'label'])] = scaler.transform(df[df.columns.difference(['attack_cat', 'label'])])
        dataX['label'] = 0
        dataX_test['label'] = 0
        dataX_abnormal['label'] = 1
        #dataX_test['label'] = np.where(dataX_test[self.label_col] == self.label_name, 0, 1)
        print("Train shape : ", dataX.shape)
        #print(dataX.head())
        print("Test normal shape : ", dataX_test.shape)
        #print(dataX_test.head())
        print("Test abnormal shape : ", dataX_abnormal.shape)
        #print(dataX_abnormal.head())
        self.input_dim = dataX.shape[1] - 1
        print("Input shape : ", self.input_dim)
        print("Latent shape : ", self.latent_dim)

        self.threshold = threshold


        if self.model_name == "ae":
            print("Using AutoEncoder model")
            if self.dataset == "Unsw":
                self.model = AutoEncoder(self.input_dim, self.latent_dim).to(device)
            else:
                self.model = AutoEncoder_Ton(self.input_dim, self.latent_dim).to(device)
            self.out_dir =  os.path.join(self.out_dir, "ae/")

        elif self.model_name == "ae2":
            print("Using AutoEncoder model")
            if self.dataset == "Unsw":
                self.model = AutoEncoder_2(self.input_dim, self.latent_dim).to(device)
            else:
                self.model = AutoEncoder_Ton2(self.input_dim, self.latent_dim).to(device)
            self.out_dir =  os.path.join(self.out_dir, "ae2/")

        elif self.model_name == "bigan":
            print("Using BiGAN model")
            if self.dataset == "Unsw": 
                self.model = BiGAN(self.input_dim, self.latent_dim).to(device)
            else:
                self.model = BiGAN_Ton(self.input_dim, self.latent_dim).to(device)
            self.out_dir =  os.path.join(self.out_dir, "bigan/")

        elif self.model_name == "bigan2":
            print("Using BiGAN 2 model")
            if self.dataset == "Unsw": 
                self.model = BiGAN2(self.input_dim, self.latent_dim).to(device)
            else:
                self.model = BiGAN_Ton2(self.input_dim, self.latent_dim).to(device)
            self.out_dir =  os.path.join(self.out_dir, "bigan2/")

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024
        print('model size: {:.3f}KB'.format(size_all_mb))
        self.out_dir = os.path.join(self.out_dir, time.strftime('%Y-%m-%d %a %H:%M:%S'))
        print(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.learning_rate = learning_rate
        self.user_fraction = fac_users # percentage of total user involed in each global training round
        total_users = self.num_clients
        print("total users: ", total_users)
        for i in range(self.num_clients):            
            id = i
            train_feature, train_label = self.get_client_data(dataX, factor=factor, i=i)
            test_feature, test_label = self.get_client_data(dataX_test, factor=test_factor, i=i)
            test_abnormal_feature, test_abnormal_label = self.get_client_data(dataX_abnormal, factor=test_abnormal_factor, i=i)

            #print(train_feature.columns)
            # Standardization
            #train_feature = self.scaler.transform(train_feature)
            #test_feature = self.scaler.transform(test_feature)

            #print(train_feature.shape)
            X_train = torch.Tensor(train_feature.values).type(torch.float32)
            y_train = torch.Tensor(train_label.values).type(torch.int32)

            X_test_normal = torch.Tensor(test_feature.values).type(torch.float32)
            y_test_normal = torch.Tensor(test_label.values).type(torch.int32)

            X_test_abnormal = torch.Tensor(test_abnormal_feature.values).type(torch.float32)
            y_test_abnormal = torch.Tensor(test_abnormal_label.values).type(torch.int32)

            train_set = [(x, y) for x, y in zip(X_train, y_train)]
            test_set = [(x, y) for x, y in zip(X_test_normal, y_test_normal)]
            test_abnormal_set = [(x, y) for x, y in zip(X_test_abnormal, y_test_abnormal)]
            #print(len(train_set))
            #print(local_epochs)
            if self.model_name == "ae" or self.model_name == "ae2":            
                user = UserAvg(device, id, train_set, test_set, test_abnormal_set,
                               (self.model, self.model_name),
                               batch_size, learning_rate, local_epochs)
            elif self.model_name == "bigan"  or self.model_name == "bigan2":
                user = UserBiGAN(device, id, train_set, test_set, test_abnormal_set,
                                 (self.model, self.model_name), self.latent_dim,
                                 batch_size, learning_rate, local_epochs)
            
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", 
              int(self.user_fraction*total_users), " / " ,total_users)
        print("Finished creating FedAvg server.")

          
    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        train_loss = test_loss = 0
        labels = errors = []
        threshold_constant = self.threshold
        if self.model_name == "ae" or self.model_name == "ae2":
            #self.threshold = 0.0005
            threshold_range = np.linspace(0.00001, 0.1, 180)
            threshold_range = np.round(threshold_range, decimals=7)
           
        elif self.model_name == "bigan" or self.model_name == "bigan2":
            #self.threshold = 5.0
            threshold_range = np.linspace(1.0, 5.0, 150)
            #threshold_range = np.linspace(2.9, 3.1, 20)
            threshold_range = np.round(threshold_range, decimals=7)

            
        for glob_iter in tqdm(range(self.num_glob_iters)):
            if(self.experiment):
                self.experiment.set_epoch(glob_iter + 1)
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each iteration
            if ((glob_iter != 0) and (glob_iter % 30 == 0)) or \
                    (glob_iter == self.num_glob_iters - 1) or \
                        (self.num_glob_iters == 1):
            #if (glob_iter == self.num_glob_iters):
                print("-------------Round number: ",glob_iter, " -------------")
                #if (glob_iter == self.num_glob_iters - 1):
                train_loss, test_loss, labels, errors = self.s_evaluate(glob_iter)
                # dictionary of lists 
                dict = {'labels': labels, 'scores': errors} 
                df = pd.DataFrame(dict)
                path = os.path.join(self.out_dir, self.prefix + "results_" + str(glob_iter) + ".csv")
                print(path)
                df.to_csv(path, index=False)
                self.export_model(glob_iter)
                #for threshold in threshold_range:
                    #print("Threshold = ", threshold)
                #    self.threshold = threshold
                    #print("Tuning threshold = ", self.threshold)
                #    train_loss, test_loss, _ , _ = self.s_evaluate(glob_iter)
                #self.threshold = threshold_constant
            else:
                print("-------------Round number: ",glob_iter, " -------------")
                train_loss, test_loss, _ , _ = self.s_evaluate(glob_iter)

            self.selected_users = self.select_users(glob_iter,
                                                    self.subusers)
            
            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                user.train(self.local_epochs)
                #print(f"selected user id: {user.id}")

            self.aggregate_parameters()



    def s_evaluate(self, glob_iter):
        
        t_ids, \
            t_tn, t_fp, t_fn, t_tp, \
                t_loss, t_ns = self.s_train_error_and_loss()

        #self.threshold = opt_t

        ids, \
            tn, fp, fn, tp, \
                loss, ano_loss, \
                      ns, labels, errors = self.s_test()  
        
        # Compute metrics
        def metrics(tn, fp, fn, tp):
            
            if tp != 0 or fp != 0 or tn != 0 or fn != 0:
                accuracy = (tp+tn)/(tn + fp + fn + tp)
            else: accuracy = 0.0

            if tp != 0 or fp != 0:
                precision = tp/(tp + fp)
            else: precision = 0.0

            if tp != 0 or fn != 0:
                recall = tp/(tp + fn)
                fnr = fn / (fn + tp)
            else: 
                recall = 0.0
                fnr = 0.0

            if precision != 0 or recall != 0 :
                f1 = 2.0 * precision * recall / (precision + recall)
            else: f1 = 0.0
            
            return accuracy, precision, recall, f1, fnr

        test_metrics = metrics(tn, fp, fn, tp)
        train_metrics = metrics(t_tn, t_fp, t_fn, t_tp)

        #print(loss)
        train_loss = np.mean(list(t_loss))

        test_loss = np.mean(list(loss))

        test_ano_loss = np.mean(list(ano_loss))

        self.rs_avg_acc.append(test_metrics[0])
        self.rs_glob_acc.append(test_metrics[0])
        self.rs_train_acc.append(train_metrics[0])
        self.rs_train_loss.append(train_loss)

        #print("stats_train[1]",stats_train[3][0])
        #print("Average Global Trainning Accurancy: ", train_acc)
        print()
        print("### Threshold = {}".format(self.threshold))
        print("Global Trainning Loss: {:.5f}".format(train_loss))
        print("Train TN: {}, FP: {}, FN: {}, TP: {}".format(t_tn, t_fp, t_fn, t_tp))                                                       
        print("Train Acc: {:.3f}, Pre: {:.3f}, Recall: {:.3f}, \
              F1: {:.3f}, FNR: {:.3f}".format(train_metrics[0],
                                            train_metrics[1],
                                            train_metrics[2],
                                            train_metrics[3],
                                            train_metrics[4]))
        print()
        print("Global Test Normal Loss: {:.5f}".format(test_loss))
        print("Global Test Abnormal Loss: {:.5f}".format(test_ano_loss))
        print("Test TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))              
        print("Test Acc: {:.3f}, Pre: {:.3f}, Recall: {:.3f}, \
               F1: {:.3f}, FNR: {:.3f}".format(test_metrics[0],
                                                test_metrics[1],
                                                test_metrics[2],
                                                test_metrics[3],
                                                test_metrics[4]))
        print()
        print_log(np.round([glob_iter,
                            self.threshold,
                            train_loss, 
                            test_loss, 
                            test_ano_loss], 5).tolist(),
                            stdout=False, 
                            fpath=os.path.join(self.out_dir, self.prefix + 'loss.txt'))
        
        print_log(np.round([glob_iter,
                            self.threshold,
                            train_metrics[0],
                            train_metrics[1], 
                            train_metrics[2], 
                            train_metrics[3], 
                            train_metrics[4]], 5).tolist(),
                            stdout=False, 
                            fpath=os.path.join(self.out_dir, self.prefix + 'train_metrics.txt')) 

        print_log(np.round([glob_iter,
                            self.threshold,
                            test_metrics[0],
                            test_metrics[1], 
                            test_metrics[2], 
                            test_metrics[3], 
                            test_metrics[4]], 5).tolist(),
                            stdout=False, 
                            fpath=os.path.join(self.out_dir, self.prefix + 'test_metrics.txt'))

        # Calculate precision, recall for various thresholds
        #precisions, recalls, thresholds = precision_recall_curve(labels, errors)
        #f1_scores = 2*recalls*precisions / (recalls + precisions)
        
        # Get the best threshold and F1 score
        #idx = np.argmax(f1_scores)
        #best_threshold, best_f1 = thresholds[idx], f1_scores[idx]
        #best_precision, best_recall = precisions[idx], recalls[idx]

        #print(f"Best F1 score: {best_f1}, Best Threshold: {best_threshold}")
        #print(labels))
        #print(errors))
        #self.plot_precision_recall_curve(labels, errors)
        #self.plot_roc_curve(labels, errors)
                                                                                                                        
                            #print("Test Accurancy: ", test_metrics[0])
        #print("Test Precision: ", test_metrics[1])
        #print("Test Recall: ", test_metrics[2])
        #print("Test F1-Score: ", test_metrics[3])
        #print("Test FNR: ", test_metrics[4])
        return train_loss, test_loss, labels, errors

    def plot_precision_recall_curve(self, true_labels, anomaly_scores):
        precisions, recalls, _ = precision_recall_curve(true_labels, anomaly_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig('PR.png', dpi=300, bbox_inches='tight')   

    def plot_roc_curve(self, true_labels, anomaly_scores):
        fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # diagonal
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic curve')
        plt.legend(loc="lower right")
        plt.savefig('ROC.png', dpi=300, bbox_inches='tight')   

    def s_test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        l_tn = 0
        l_fp = 0
        l_fn = 0
        l_tp = 0
        losses = []
        ano_losses = []
        mean_accurancy = []
        total_preds = []
        total_labels = []
        total_errors = []
        for c in self.users:
            tn, fp, fn, tp, loss, ano_loss, \
                preds, c_labels, ns, errs = c.c_test(self.threshold)
            l_tn += tn
            l_fp += fp
            l_fn += fn
            l_tp += tp
            num_samples.append(ns)
            losses.append(loss*1.0)
            ano_losses.append(ano_loss * 1.0)
            total_preds.extend(preds)
            total_labels.extend(c_labels)
            total_errors.extend(errs)
            #mean_accurancy.append(ma)
        ids = [c.id for c in self.users]

        #print(total_labels)
        
        return ids, l_tn, l_fp, l_fn, l_tp, losses, ano_losses, num_samples,\
                total_labels, total_errors \
                   
    def s_train_error_and_loss(self):
        num_samples = []
        l_tn = 0
        l_fp = 0
        l_fn = 0
        l_tp = 0
        opt_t = 0
        tot_correct = []
        losses = []
        for c in self.users:
            tn, fp, fn, tp, loss, ns = c.c_train_error_and_loss(self.threshold) 
            l_tn += tn
            l_fp += fp
            l_fn += fn
            l_tp += tp
            num_samples.append(ns)
            losses.append(loss*1.0)

        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, l_tn, l_fp, l_fn, l_tp, losses, num_samples
    
        # self.save_results()
        #model_name = f"FedAVG{self.dataset}_num_user_{self.total_users}_globalEpochs{self.num_glob_iters}_localEpochs{self.local_epochs}_fac{self.subusers}"
        # if self.mulTS == 0:
        #    self.save_model_lstm(model_name)
        #else:
        #    self.save_model_lstm_mulTS(model_name)


    '''
    Get data from csv file
    '''
    def get_data(self, i):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"client{i+1}_preprocessed.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read preprocessed data from csv file
        client_train = pd.read_csv(client_path)
        client_train = client_train.to_numpy()

        return client_train

    '''
    Get data from kdd dataset (.csv file)
    '''
    def get_data_kdd(self, i):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"client{i+1}_kdd_std.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read preprocessed data from csv file
        client_train = pd.read_csv(client_path)
        client_train = client_train.to_numpy()

        return client_train
    
    '''
    Get data from nsl kdd dataset (.csv file)
    '''
    def get_data_nsl_kdd(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"nslkdd_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read data from csv file and create non-i.i.d data for each client
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['dst_bytes'])
        client_train = client_train.drop(['Unnamed: 0', 'outcome'], axis=1)
        print(client_train['dst_bytes'])
        print("Sorted!!!!!")

        return client_train

    '''
    Get data from unsw nb15 dataset (.csv file)
    '''
    def get_data_unsw_nb15(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        test_data_path = os.path.join(directory, "abnormal_detection_data/test")
        print(data_path)
        file_name = f"unswnb15_train_normal.csv"
        client_path = os.path.join(data_path, file_name)


        # Read data from csv file and create non-i.i.d data for each client
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['ct_srv_src'])
        #print(client_train.head())
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        # print(client_train['ct_srv_src'])

        test_name = f"unsw_nb15_test_nocat.csv"
        test_normal_name = f"unswnb15_test_normal_full.csv"
        test_abnormal_name = f"unswnb15_test_abnormal.csv"
        test_path = os.path.join(test_data_path, test_name)
        test_normal_path = os.path.join(test_data_path, test_normal_name)
        test_abnormal_path = os.path.join(test_data_path, test_abnormal_name)
        #print(client_path)
        #file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/unswnb15_test_normal.csv")
        #file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/unswnb15_test_abnormal.csv")

        #client_test = pd.read_csv(test_path)
        #client_normal_test_full = client_test.loc[client_test['attack_cat'] == 'Normal']
        #client_normal_test_full = client_normal_test_full.drop(['attack_cat'], axis=1)
        #print(client_normal_test_full)
        #client_normal_test_full.to_csv(test_data_path + '/unswnb15_test_normal_full.csv', index=False)

        client_normal_test = pd.read_csv(test_normal_path)
        client_normal_test = client_normal_test.drop(["Unnamed: 0"], axis=1)

        #print(client_normal_test.shape)
        client_normal_test_sample = client_normal_test[:20000]
        client_normal_test_sample = client_normal_test_sample.sort_values(by=['ct_srv_src'])

        client_abnormal_test = pd.read_csv(test_abnormal_path)
        client_abnormal_test = client_abnormal_test.drop(["Unnamed: 0"], axis=1)
        client_abnormal_test = client_abnormal_test.sort_values(by=['ct_srv_src'])

        #print(client_normal_test_sample.head())
        #print(client_abnormal_test.head())
        #client_test = client_test.drop(["Unnamed: 0"], axis=1)
        print("Created Non-iid Data!!!!!")

        return client_train, client_normal_test_sample, client_abnormal_test

    '''
    Get data from ToN dataset (.csv file)
    '''
    def get_data_Ton(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"ton_train_normal_49.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read data from csv file and create non-i.i.d data for each client
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['src_port'])
        #client_train = client_train.drop(["Unnamed: 0"], axis=1)

        # print(client_train['dns_qtype'])

        test_data_path = os.path.join(directory, "abnormal_detection_data/test")
        #test_name = f"unsw_nb15_test_nocat.csv"
        test_normal_name = f"ton_test_normal_49.csv"
        test_abnormal_name = f"ton_test_abnormal_49.csv"
        #test_path = os.path.join(test_data_path, test_name)
        test_normal_path = os.path.join(test_data_path, test_normal_name)
        test_abnormal_path = os.path.join(test_data_path, test_abnormal_name)
        #print(client_path)
        #file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/unswnb15_test_normal.csv")
        #file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/unswnb15_test_abnormal.csv")

        #client_test = pd.read_csv(test_path)
        #client_normal_test_full = client_test.loc[client_test['attack_cat'] == 'Normal']
        #client_normal_test_full = client_normal_test_full.drop(['attack_cat'], axis=1)
        #print(client_normal_test_full)
        #client_normal_test_full.to_csv(test_data_path + '/unswnb15_test_normal_full.csv', index=False)

        client_normal_test = pd.read_csv(test_normal_path)
        #client_normal_test = client_normal_test.drop(["Unnamed: 0"], axis=1)

        #print(client_normal_test.shape)
        client_normal_test_sample = client_normal_test[:10000]
        client_normal_test_sample = client_normal_test_sample.sort_values(by=['src_port'])

        client_abnormal_test = pd.read_csv(test_abnormal_path)
        #client_abnormal_test = client_abnormal_test.drop(["Unnamed: 0"], axis=1)
        client_abnormal_test = client_abnormal_test.sort_values(by=['src_port'])
     
        print("Created Non-iid Data!!!!!")

        return client_train, client_normal_test_sample, client_abnormal_test

    '''
    Preprocessing data step
    '''
    def prep_data(self, dataX):
        change_dataX = dataX.copy()
        featuresToScale = change_dataX.columns
        sX = StandardScaler(copy=True)
        change_dataX.loc[:,featuresToScale] = sX.fit_transform(change_dataX[featuresToScale])
        return change_dataX

    '''
    Divide data to clients
    '''
    def get_client_data(self, data, factor, i):
        # Read data frame for each client
        factor = int(factor)
        client_data_i = data[factor*i:factor*(i+1)].copy()
        # Preprocess data
        #client_data = self.prep_data(dataX)

        train_feature = client_data_i[client_data_i.columns.difference([self.label_col, 'label'])]
        train_label = client_data_i['label']

        #print(train_label.value_counts())
        #client_data = client_data_i.to_numpy()
        return train_feature, train_label
    
    def export_model(self, glob_iter = 30):
        #model_path = os.path.join(self.out_dir, self.prefix + 'loss.txt'))
        model_path = self.out_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, self.prefix + "model_" + str(glob_iter) + ".pth"))