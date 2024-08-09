

# Fed PCA: Federated PCA on Grassmann Manifold for Anomaly Detection in IoT Networks [IEEE Transactions on Networking]
This repository is for the Experiment Section of the paper: "Federated PCA on Grassmann Manifold for IoT Anomaly Detection"

Authors: Tung-Anh Nguyen, Long Tan Le, Tuan Dung Nguyen, Wei Bao, Suranga Seneviratne, Choong Seon Hong, Nguyen H. Tran

Link paper:
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10593810
Part of this work has been accepted at INFOCOM 2023 (Link paper: https://arxiv.org/pdf/2212.12121.pdf)


# Software requirements:
- numpy, sklearn, pytorch, matplotlib.

- To download the dependencies: pip3 install -r requirements.txt

- The code can be run on any pc or google colab.
  
# Dataset:
- UNSW-NB15
- ToN-IoT

# Experimental results
## UNSW-NB15 ##
| Algorithm | Rank k | Accuracy     | Precision     | Recall       | F1-Score     | FN             | Global Iteration to Converge | Time to Converge |
|:---------:|:------:|:--------:    |:---------:    |:--------:    |:--------:    |:--------------:|:------------:                |:----------------:|
| `FedPG`   | 2      | 90.0046%     | 89.9895%      | **98.7757%** | 94.1781%     | **1.2243%**    | **20**                       | **5.855s**       |
| `FedPE`   | 2      | **90.2172%** | **90.4981%**  | 98.3897%     | **94.2791%** | 1.6103%        | 200                          | 63.198s          |

## ToN-IoT ##
| Algorithm | Rank k | Accuracy     | Precision     | Recall         | F1-Score      | FN             | Global Iteration to Converge | Time to Converge |
|:---------:|:------:|:--------:    |:---------:    |:--------:      |:--------:     |:--------------:|:------------:                |:----------------:|
| `FedPG`   | 2      | **91.0227%** | **88.3039%**  | **100.000%**   | **93.7887%**  | **0.0000%**    | **20**                       | **4.551s**       |
| `FedPE`   | 2      | 91.0066%     | 88.2854%      | 100.000%       | 93.7783%      | 0.0000%        | 200                          | 54.0505          |

# Commands to run multiple experiments
## Run experiments for both datasets using scripts

### Run all experiments
<pre></code>
bash all_experiments.sh
<code></pre>

### Experiments on Global Iteration
<pre></code>
bash global_iteration_experiments_script.sh
<code></pre>

### Experiments on Local Iteration
<pre></code>
bash local_iteration_experiments_script.sh
<code></pre>

### Experiments on Rank-k
<pre></code>
bash rank_experiments_script.sh
<code></pre>

### Experiments on N_clients
<pre></code>
bash clients_experiments_script.sh
<code></pre>

# Commands to run an experiment

## UNSW-NB15 
### Experiments on Global Iteration

#### FedPG
<pre></code>
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 50 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 70 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 90 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 110 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 130 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 150 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 170 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 190 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPG --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
<code></pre>

#### FedPE
<pre></code>
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 20 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 30 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 40 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 50 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 60 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 70 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 80 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 90 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 100 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 110 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 120 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 130 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 140 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 150 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 160 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 170 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 180 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 190 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
python3 main.py --algorithm FedPE --dataset Unsw --learning_rate 0.0001 --num_global_iters 200 --dim 2 --clients 20 --subusers 0.1 --local_epochs 30 --exp_type Global_iter
<code></pre>

<pre></code>
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
<code></pre>

### Experiments on Rank-k

#### FedPG
<pre></code>
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
<code></pre>

#### FedPE
<pre></code>
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
<code></pre>

### Experiments on N_clients

#### FedPG
<pre></code>
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
<code></pre>

#### FedPE
<pre></code>
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
<code></pre>
