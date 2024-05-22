import csv 
import os 
import time
# Setup contents
header = ['N_clients', 'Global_iter', 'Local_iter', 'Rank', 'Loss', 'Acc', 'Precision', 'Recall', 'F1-score', 'FN', 'Time']

def metrics_exp_store(file_path, data_row):
    data = data_row
    HEADER = True

    # Check file existence
    if os.path.exists(file_path) == False:
        # Create file if it does not exist

        with open(file_path, 'w') as f:
            HEADER = False

    with open(file_path, 'a') as f:
        writer = csv.writer(f)
        if HEADER:
            writer.writerow(data)
        else:
            writer.writerow(header)
            writer.writerow(data)
            HEADER = True 


def print_log(message, fpath=None, stdout=True, print_time=False):
  if print_time:
    timestr = time.strftime('%Y-%m-%d %a %H:%M:%S')
    message = f'{timestr} | {message}'
  if stdout:
    print(message)
  if fpath is not None:
    with open(fpath, 'a') as f:
      print(message, file=f)