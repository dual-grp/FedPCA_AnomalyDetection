echo "Run experiments for Global Iterations ..." 
bash global_iteration_experiments_script.sh
echo "Run experiments for Local Iterations ..." 
bash local_iteration_experiments_script.sh
echo "Run experiments for Rank-k ..."
bash rank_experiments_script.sh
echo "Run experiments for N_clients..."
bash clients_experiments_script.sh
