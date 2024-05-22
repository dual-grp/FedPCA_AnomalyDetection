#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="KDD", choices=["Unsw", "Iot23", "Ton"])
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default = 0.000001, help="Local learning rate")
    parser.add_argument("--ro", type=float, default=0.01, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=250)
    parser.add_argument("--local_epochs", type=int, default = 30)
    parser.add_argument("--dim", type=int, default = 3)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="ADMM",choices=["FedPG","FedPE", "FedAE", "FedAE2", "FedBiGAN", "FedBiGAN2"])
    parser.add_argument("--clients", type = int, default = 1, help="Total number of Clients")
    parser.add_argument("--subusers", type = float, default = 1, help="Number of Users per round")
    parser.add_argument("--threshold", type = float, default = 0.5, help="Algorithms threshold")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=0, help="log data to commet")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments")
    parser.add_argument("--exp_type", type=str, default="",choices=["Global_iter","Rank_k", "N_clients", "Local_iter"])

    args = parser.parse_args()

    return args
