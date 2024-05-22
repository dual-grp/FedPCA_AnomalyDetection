'''
Import necessary libraries
'''
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from utils.pca_utils import *
from sklearn.metrics import classification_report, confusion_matrix

'''
Data preprocessing 
'''

MIN_MAX = 1
STANDAR = 2
ROBUST = 3

# Choose the preprocessing methods:
def prep_data(dataX, prep_type=1):

    change_dataX = dataX.copy()
    featuresToScale = change_dataX.columns

    if prep_type == MIN_MAX:
        min_max = MinMaxScaler()
        change_dataX.loc[:,featuresToScale] = min_max.fit_transform(change_dataX[featuresToScale])
    elif prep_type == STANDAR:
        sX = StandardScaler(copy=True)
        change_dataX.loc[:,featuresToScale] = sX.fit_transform(change_dataX[featuresToScale])
    else:
        robScaler = RobustScaler()
        change_dataX.loc[:,featuresToScale] = robScaler.fit_transform(change_dataX[featuresToScale])

    return change_dataX


'''
 Define the score function for abnormal detection
'''
def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF) - np.array(reducedDF))**2, axis=1) 
    return loss


'''
Function test on all types of attacks from csv file
'''
def test_on_random_attack_samples_csv(V_k, thres_hold = 6, prep_type=MIN_MAX):
    dataX = pd.read_csv('abnormal_detection_data/test/abnormal_test_34_fea.csv', index_col=False)
    dataX = dataX.drop(['outcome', 'Unnamed: 0'], axis=1)
    dataX = prep_data(dataX, prep_type=STANDAR)
    data_transform = self_pca_transform_with_zero_mean(dataX, V_k)
    data_inverse = self_inverse_transform_with_zero_mean(data_transform, V_k)
    dataX = prep_data(dataX, prep_type=MIN_MAX)
    data_inverse = prep_data(data_inverse, prep_type=MIN_MAX)
    abnormal_score = anomalyScores(dataX, data_inverse)
    index = abnormal_score < thres_hold
    num_samples = dataX.shape[0]
    total_misclassified_samples = len(abnormal_score[index])
    return num_samples, total_misclassified_samples

'''
Test on another 10000 sample of normal data get data from csv file
'''
def test_on_normal_data_samples_csv(V_k, thres_hold=6):
    normal_data_10000 = pd.read_csv('abnormal_detection_data/test/kdd_10000_34_fea.csv', index_col=False)
    # Standardization 
    normal_data_10000 = prep_data(normal_data_10000, prep_type=STANDAR)
    # Transform data using Federated PCA components
    normal_data_pca = self_pca_transform_with_zero_mean(normal_data_10000, V_k)
    normal_data_inverse = self_inverse_transform_with_zero_mean(normal_data_pca, V_k)
    # Min max normalization before testing to keep balance between features
    normal_data_10000 = prep_data(normal_data_10000, prep_type=MIN_MAX)
    normal_data_inverse = prep_data(normal_data_inverse, prep_type=MIN_MAX)
    # Get anomalous score
    abnormal_score = anomalyScores(normal_data_10000, normal_data_inverse)
    # Get the misclassified samples
    index = abnormal_score > thres_hold
    total_normal_samples = normal_data_10000.shape[0]
    return total_normal_samples, len(abnormal_score[index])
    
'''
Test on both normal and abnormal data and calculate the F-Score
'''
def kdd_test(V_k, thres_hold):
    # Get total samples and misclassified samples on normal data
    normal_total_samples, normal_mis_samples = test_on_normal_data_samples_csv(V_k, thres_hold=thres_hold)
    FN = normal_mis_samples
    TN = normal_total_samples - normal_mis_samples
    # Get total samples and misclassified samples on abnormal data
    abnormal_total_samples, abnormal_mis_samples = test_on_random_attack_samples_csv(V_k, thres_hold=thres_hold)
    FP = abnormal_mis_samples
    TP = abnormal_total_samples - abnormal_mis_samples
    # Estimate performance based on the F-Score
    precision_score = TP/(FP + TP)
    recall_score = TP/(FN + TP)
    accuracy_score = (TP + TN)/ (TP + FN + TN + FP)
    f1_score = 2*precision_score*recall_score/(precision_score + recall_score)
    print(f"Precision: {precision_score * 100.0}")
    print(f"Recall: {recall_score * 100.0}")
    print(f"Accuracy score: {accuracy_score * 100.0}")
    print(f"F1 score: {f1_score * 100.0}")
    return accuracy_score*100.0

# Define the score function for abnormal detection which scale abnormal score in range [0:1]
def anomalyScores_0_1(originalDF, reducedDF):
  loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
  loss = pd.Series(data=loss,index=originalDF.index)
  loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
  return loss

# Calculate F-Score 
def results_analysis(df_gt_score, threshold, log=0):
  df_gt_pred = pd.DataFrame()
  df_gt_pred['ground_true'] = df_gt_score['ground_true']
  index = df_gt_score['anomalyScore'] > threshold
  df_gt_pred['prediction'] = index.astype(int)

  TN, FP, FN, TP = confusion_matrix(df_gt_pred['ground_true'], df_gt_pred['prediction']).ravel()
  precision_score = TP/(FP + TP)
  recall_score = TP/(FN + TP)
  accuracy_score = (TP + TN)/ (TP + FN + TN + FP)
  f1_score = 2*precision_score*recall_score/(precision_score + recall_score)
  fpr = FP / (FP+TN) # False positive rate (FPR); False Alarm = FP/N
  fng = FN / (TP+FN)

  if log:
    print(f"Precision: {np.round(precision_score * 100.0,4)}%")
    print(f"Recall: {np.round(recall_score * 100.0,4)}%")
    print(f"Accuracy score: {np.round(accuracy_score * 100.0,4)}%")
    print(f"F1 score: {np.round(f1_score * 100.0,4)}%")
    print(f"False alarm: {np.round(fpr * 100.0,4)}%")
    print(f"False Negative: {np.round(fng * 100.0,4)}%")

  return precision_score, recall_score, accuracy_score, f1_score, fpr, fng

# Test on NSL_KDD_Test+
def nsl_kdd_test(V_k):
  # Read data from csv files
  file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/nslkdd_train_normal.csv")
  file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/nslkdd_test_normal.csv")
  file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/nslkdd_test_abnormal.csv")
  df_train = pd.read_csv(file_path_train, index_col = 0)
  df_test_normal = pd.read_csv(file_path_test_normal, index_col = 0)
  df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col = 0)
  df_train = df_train.drop('outcome', axis=1)
  df_test_normal = df_test_normal.drop('outcome', axis=1)
  df_test_abnormal = df_test_abnormal.drop('outcome', axis=1)
  df_test = pd.concat([df_test_normal, df_test_abnormal])
  df_test.columns = df_test_abnormal.columns

  # Standardization over Testing
  scaler = StandardScaler()
  scaler.fit(df_train)
  df_test = pd.DataFrame(scaler.transform(df_test))
  df_test.columns = df_test_abnormal.columns
  
  # FedPCA: Obtain dataframe of groundTrue and anomalyScore using V_k
  df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k)
  df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k)
  abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
  df_gt_score = pd.DataFrame(); df_gt_pred = pd.DataFrame()
  df_gt_score['ground_true'] = np.concatenate([np.zeros(len(df_test_normal)), np.ones(len(df_test_abnormal))])
  df_gt_score['anomalyScore'] = abnormal_score

  # Get results analysis
  precision_score, recall_score, accuracy_score, f1_score, fpr, fng =  results_analysis(df_gt_score, threshold=0.00025, log=1)

  return accuracy_score

# Test on UNSW NB15
def unsw_nb15_test(V_k):
  # Read data from csv files
  file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/unswnb15_train_normal.csv")
  file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/unswnb15_test_normal_full.csv")
  file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/unswnb15_test_abnormal.csv")
  df_train = pd.read_csv(file_path_train, index_col = 0)
  df_test_normal = pd.read_csv(file_path_test_normal, index_col = 0)
  df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col = 0)
  # print(df_test_normal.shape)

#   df_train = df_train.drop("Unnamed: 0", axis=1)
#   df_test_normal = df_test_normal.drop("Unnamed: 0", axis=1)
#   df_test_abnormal = df_test_abnormal.drop("Unnamed: 0", axis=1)
  df_test_normal = df_test_normal[:20000]
  df_test = pd.concat([df_test_normal, df_test_abnormal])
  df_test.columns = df_test_abnormal.columns

  # Standardization over Testing
  scaler = StandardScaler()
  scaler.fit(df_train)
  df_test = pd.DataFrame(scaler.transform(df_test))
  df_test.columns = df_test_abnormal.columns
  
  # FedPCA: Obtain dataframe of groundTrue and anomalyScore using V_k
  df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k)
  df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k)
  abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
  df_gt_score = pd.DataFrame(); df_gt_pred = pd.DataFrame()
  df_gt_score['ground_true'] = np.concatenate([np.zeros(len(df_test_normal)), np.ones(len(df_test_abnormal))])
  df_gt_score['anomalyScore'] = abnormal_score

  # Get results analysis
  precision_score, recall_score, accuracy_score, f1_score, fpr, fng =  results_analysis(df_gt_score, threshold=0.000006, log=1)

  return precision_score, recall_score, accuracy_score, f1_score, fpr, fng


# Test on IoT23
def iot23_test(V_k):
  # Read data from csv files
  file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/iot23_train_normal.csv")
  file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/iot23_test_normal.csv")
  file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/iot23_test_abnormal.csv")
  df_train = pd.read_csv(file_path_train, index_col = 0)
  df_test_normal = pd.read_csv(file_path_test_normal, index_col = 0)
  df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col = 0)
#   df_train = df_train.drop("Unnamed: 0", axis=1)
#   df_test_normal = df_test_normal.drop("Unnamed: 0", axis=1)
#   df_test_abnormal = df_test_abnormal.drop("Unnamed: 0", axis=1)
  # df_test_normal = df_test_normal[:5000]
  df_test = pd.concat([df_test_normal, df_test_abnormal])
  df_test.columns = df_test_abnormal.columns

  # Standardization over Testing
  scaler = StandardScaler()
  scaler.fit(df_train)
  df_test = pd.DataFrame(scaler.transform(df_test))
  df_test.columns = df_test_abnormal.columns
  
  # FedPCA: Obtain dataframe of groundTrue and anomalyScore using V_k
  df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k)
  df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k)
  abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
  df_gt_score = pd.DataFrame(); df_gt_pred = pd.DataFrame()
  df_gt_score['ground_true'] = np.concatenate([np.zeros(len(df_test_normal)), np.ones(len(df_test_abnormal))])
  df_gt_score['anomalyScore'] = abnormal_score

  # Get results analysis
  precision_score, recall_score, accuracy_score, f1_score, fpr, fng =  results_analysis(df_gt_score, threshold=1e-13, log=1)

  return precision_score, recall_score, accuracy_score, f1_score, fpr, fng


# Test on ToN IoT
def ton_test(V_k):
  # Read data from csv files
  file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/ton_train_normal_49.csv")
  file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/ton_test_normal_49.csv")
  file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/ton_test_abnormal_49.csv")
  df_train = pd.read_csv(file_path_train, index_col = 0)
  df_test_normal = pd.read_csv(file_path_test_normal, index_col = 0)
  df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col = 0)
#   df_train = df_train.drop("Unnamed: 0", axis=1)
#   df_test_normal = df_test_normal.drop("Unnamed: 0", axis=1)
#   df_test_abnormal = df_test_abnormal.drop("Unnamed: 0", axis=1)
  print(df_test_abnormal.shape)
  df_test_normal = df_test_normal[:10000]
  
  df_test = pd.concat([df_test_normal, df_test_abnormal])
  df_test.columns = df_test_abnormal.columns

  # Standardization over Testing
  scaler = StandardScaler()
  scaler.fit(df_train)
  df_test = pd.DataFrame(scaler.transform(df_test))
  df_test.columns = df_test_abnormal.columns
  
  # FedPCA: Obtain dataframe of groundTrue and anomalyScore using V_k
  df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k)
  df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k)
  abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
  df_gt_score = pd.DataFrame(); df_gt_pred = pd.DataFrame()
  df_gt_score['ground_true'] = np.concatenate([np.zeros(len(df_test_normal)), np.ones(len(df_test_abnormal))])
  df_gt_score['anomalyScore'] = abnormal_score

  # Get results analysis
  precision_score, recall_score, accuracy_score, f1_score, fpr, fng =  results_analysis(df_gt_score, threshold=3e-8, log=1)

  return precision_score, recall_score, accuracy_score, f1_score, fpr, fng