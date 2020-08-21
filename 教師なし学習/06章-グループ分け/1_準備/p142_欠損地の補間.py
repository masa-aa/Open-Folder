# Import libraries
'''Main and Data Viz and Data Prep and Model Evaluation and Algorithms'''
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, cophenet, fcluster
import fastcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn import preprocessing as pp
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import re
import pickle
import gzip
from sklearn.impute import SimpleImputer  # versionによってはpp.Imputerが存在しない


color = sns.color_palette()
# Load the data
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'lending_club_data', 'LoanStats3a.csv'])
data = pd.read_csv(current_path + file)

# columusを削る(不要なデータを削除 特徴量145->37)
columnsToKeep = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term',
                 'int_rate', 'installment', 'grade', 'sub_grade',
                 'emp_length', 'home_ownership', 'annual_inc',
                 'verification_status', 'pymnt_plan', 'purpose',
                 'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
                 'mths_since_last_delinq', 'mths_since_last_record',
                 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                 'total_acc', 'initial_list_status', 'out_prncp',
                 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
                 'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
                 'last_pymnt_amnt']

data = data.loc[:, columnsToKeep]
# print(data.shape) # (42542,37)
# print(data.head)

# Transform features from string to numeric
for i in ["term", "int_rate", "emp_length", "revol_util"]:
    data.loc[:, i] = \
        data.loc[:, i].apply(lambda x: re.sub("[^0-9]", "", str(x)))  # 0-9以外を""に置換する
    # apply(f):fを各列に対して適用, applymap(f):fを全体に適用
    data.loc[:, i] = pd.to_numeric(data.loc[:, i])  # floatに変換 文字を含むとNaN


# Determine which features are numerical
numericalFeats = [x for x in data.columns if data[x].dtype != 'object']

# Display NaNs by feature
# nanCounter = np.isnan(data.loc[:, numericalFeats]).sum()
# print(nanCounter)


# Impute NaNs with mean
fillWithMean = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term',
                'int_rate', 'installment', 'emp_length', 'annual_inc',
                'dti', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
                'out_prncp', 'out_prncp_inv', 'total_pymnt',
                'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
                'last_pymnt_amnt']

# Impute NaNs with zero
fillWithZero = ['delinq_2yrs', 'mths_since_last_delinq',
                'mths_since_last_record', 'pub_rec', 'total_rec_late_fee',
                'recoveries', 'collection_recovery_fee']

# Perform imputation
im = SimpleImputer(strategy='mean')  # NaNをmeanに変えるやつ
data.loc[:, fillWithMean] = im.fit_transform(data[fillWithMean])

data.loc[:, fillWithZero] = data.loc[:, fillWithZero].fillna(value=0, axis=1)  # NaNを0に変換

# Check for NaNs one last time
nanCounter = np.isnan(data.loc[:, numericalFeats]).sum()
print(nanCounter)
