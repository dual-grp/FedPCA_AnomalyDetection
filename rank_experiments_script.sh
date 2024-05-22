#!/bin/bash

# Experiments on Rank_k
# ## UNSW-NB15

# ### FedPG
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 6 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 10 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 18 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 22 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 26 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 30 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 34 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 39 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k

# ### FedPE
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 6 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 10 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 18 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 22 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 26 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 30 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 34 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
# python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 39 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k

##########################################################################################################################################################################
## ToN-IoT

# ### FedPG
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 4 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 6 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 8 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 10 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 12 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 16 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 18 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 23 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 25 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 27 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 29 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 31 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 33 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 35 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 37 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 39 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 41 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 43 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 45 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 47 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPG --dataset Ton --learning_rate 0.0001 --num_global_iters 30 --dim 49 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k

### FedPE
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 4 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 6 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 8 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 10 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 12 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 14 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 16 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 18 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 23 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 25 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 27 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 29 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 31 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 33 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 35 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 37 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 39 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 41 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 43 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 45 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 47 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k
python3 main.py --algorithm FedPE --dataset Ton --learning_rate 0.0001 --num_global_iters 200 --dim 49 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Rank_k