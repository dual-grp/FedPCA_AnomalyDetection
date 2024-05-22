#!/bin/bash

# Local Epoch
python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.01 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 64 --subusers $3 --threshold 0.02 --local_epochs 10 --gpu $4

python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.01 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 64 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4

python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.01 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 64 --subusers $3 --threshold 0.02 --local_epochs 30 --gpu $4

# Learning Rate
python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.01 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 64 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4

python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.05 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 64 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4

python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.1 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 64 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4

# Batch Size
python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.1 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 64 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4

python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.1 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 128 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4

python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.1 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 256 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4


# Latent dim
python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.1 --num_global_iters 100 \
                --clients $2 --dim 8 --batch_size 128 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4

python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.1 --num_global_iters 100 \
                --clients $2 --dim 16 --batch_size 128 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4

python3 main.py --algorithm FedBiGAN --dataset $1 --learning_rate 0.1 --num_global_iters 100 \
                --clients $2 --dim 20 --batch_size 128 --subusers $3 --threshold 0.02 --local_epochs 20 --gpu $4