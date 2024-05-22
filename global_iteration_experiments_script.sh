#!/bin/bash


# Experiments on Global Iterations
##UNSW-NB15

# ### FedPG
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 50 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 70 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 90 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 110 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 130 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 150 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 170 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 190 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter

# ### FedPE
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 50 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 70 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 90 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 110 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 130 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 150 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 170 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 190 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter

# ##########################################################################################################################################################################

## ToN-IoT

# ### FedPE
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 50 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 70 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 90 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 110 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 130 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 150 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 170 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 190 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
# python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter

### FedPG
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 50 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 70 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 90 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 110 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 130 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 150 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 170 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 190 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter

