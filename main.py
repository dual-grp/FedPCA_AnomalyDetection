#!/usr/bin/env python
#from comet_ml import Experiment
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravg import ServerAvg
from FLAlgorithms.servers.serverAbnormalDetection import AbnormalDetection
from utils.model_utils import read_data
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
from utils.options import args_parser

# import comet_ml at the top of your file
#                                                                                                                           
# Create an experiment with your api key:
def main(experiment, dataset, algorithm, batch_size, learning_rate, ro, num_glob_iters,
         local_epochs, clients, numusers,dim, threshold, times, gpu, exp_type):
    
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    data = dataset
    if algorithm == "FedAE":
        model = "ae"
        #print(numusers)
        server = ServerAvg(algorithm, experiment, device, data, 
                           model, batch_size, learning_rate, 
                           num_glob_iters, local_epochs, dim, clients, numusers, threshold, times)
    elif algorithm == "FedAE2":
        model = "ae2"
        #print(numusers)
        server = ServerAvg(algorithm, experiment, device, data, 
                           model, batch_size, learning_rate, 
                           num_glob_iters, local_epochs, dim, clients, numusers, threshold, times)
                
    elif algorithm == "FedBiGAN":
        model = "bigan"
        server = ServerAvg(algorithm, experiment, device, data, 
                           model, batch_size, learning_rate, 
                           num_glob_iters, local_epochs, dim, clients, numusers, threshold, times)    
    elif algorithm == "FedBiGAN2":
        model = "bigan2"
        server = ServerAvg(algorithm, experiment, device, data, 
                           model, batch_size, learning_rate, 
                           num_glob_iters, local_epochs, dim, clients, numusers, threshold, times)   
    else:
        server = AbnormalDetection(algorithm, experiment, device, data, learning_rate, ro, num_glob_iters, local_epochs, clients, numusers, dim, times, exp_type)
    server.train()

if __name__ == "__main__":
    args = args_parser()
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.ro))
    print("Subset of users      : {}".format(args.subusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    # print("Dataset       : KDD")
    print("Dataset       : {}".format(args.dataset))
    print("=" * 80)

    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="VtHmmkcG2ngy1isOwjkm5sHhP",
            project_name="multitask-for-test",
            workspace="federated-learning-exp",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "ro":args.ro,
            "dim" : args.dim,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "clients":args.clients,
            "numusers": args.subusers,
            "threshold": args.threshold,
            "times" : args.times,
            "gpu": args.gpu,
            "cut-off": args.cutoff
        }
        
        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ro = args.ro,   
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        clients = args.clients,
        numusers = args.subusers,
        dim = args.dim,
        threshold = args.threshold,
        times = args.times,
        gpu=args.gpu,
        exp_type=args.exp_type
        )


