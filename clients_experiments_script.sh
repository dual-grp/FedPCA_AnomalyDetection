#!/bin/bash

# Experiments on N_clients

## UNSW-NB15

### FedPG
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 100 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 200 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 300 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 400 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 500 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 600 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 700 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 800 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 900 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 1000 --subusers 0.1 --local_epochs 30 --exp_type N_clients

### FedPE
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 100 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 200 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 300 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 400 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 500 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 600 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 700 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 800 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 900 --subusers 0.1 --local_epochs 30 --exp_type N_clients
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 1000 --subusers 0.1 --local_epochs 30 --exp_type N_clients

##########################################################################################################################################################################
## ToN-IoT

### FedPG
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 100 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 200 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 300 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 400 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 500 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 600 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 700 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 800 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 900 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 1000 --subusers 0.1 --local_epochs 30 --exp_type N_clients

### FedPE
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 100 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 200 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 300 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 400 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 500 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 600 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 700 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 800 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 900 --subusers 0.1 --local_epochs 30 --exp_type N_clients
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 1000 --subusers 0.1 --local_epochs 30 --exp_type N_clients