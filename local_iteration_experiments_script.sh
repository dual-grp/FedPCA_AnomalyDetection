#!/bin/bash

# Experiments on Local_iter

## UNSW-NB15

### FedPG

# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter

### FedPE

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

##########################################################################################################################################################################

## ToN-IoT

### FedPG

python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter

### FedPE

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter

python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 10 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 20 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 40 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 50 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 60 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 70 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 80 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 90 --exp_type Local_iter
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 100 --exp_type Local_iter