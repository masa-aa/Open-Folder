# Import libraries
'''Main and Data Viz and Data Prep and Model Evaluation and Algorithms'''
import csv
import hdbscan
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram
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
# nanCounter = np.isnan(data.loc[:, numericalFeats]).sum()
# print(nanCounter)


# 特徴量を追加
data['installmentOverLoanAmnt'] = data.installment / data.loan_amnt
data['loanAmntOverIncome'] = data.loan_amnt / data.annual_inc
data['revol_balOverIncome'] = data.revol_bal / data.annual_inc
data['totalPymntOverIncome'] = data.total_pymnt / data.annual_inc
data['totalPymntInvOverIncome'] = data.total_pymnt_inv / data.annual_inc
data['totalRecPrncpOverIncome'] = data.total_rec_prncp / data.annual_inc
data['totalRecIncOverIncome'] = data.total_rec_int / data.annual_inc

newFeats = ['installmentOverLoanAmnt', 'loanAmntOverIncome',
            'revol_balOverIncome', 'totalPymntOverIncome',
            'totalPymntInvOverIncome', 'totalRecPrncpOverIncome',
            'totalRecIncOverIncome']

# Select features for training
numericalPlusNewFeats = numericalFeats + newFeats
X_train = data.loc[:, numericalPlusNewFeats]

# スケール変換
sX = pp.StandardScaler()
X_train.loc[:, :] = sX.fit_transform(X_train)


# 評価用ラベル
labels = data.grade
labels.unique()


# ローングレードのNaNをZに変換
labels = labels.fillna(value="Z")
# Convert labels to numerical values
lbl = pp.LabelEncoder()  # 符号化（エンコードする), 座圧
lbl.fit(list(labels.values))
labels = pd.Series(data=lbl.transform(labels.values), name="grade")
# Store as y_train
y_train = labels

labelsOriginalVSNew = pd.concat([labels, data.grade], axis=1)  # 連結
# print(labelsOriginalVSNew)


# グレードが下がるにつれ利子率が上がることを確認する.
interestAndGrade = pd.DataFrame(data=[data.int_rate, labels])
interestAndGrade = interestAndGrade.T

# print(interestAndGrade.groupby("grade").mean())

# クラスタリングの評価関数 いつもの


def analyzeCluster(clusterDF, labelsDF):
    countByCluster = \
        pd.DataFrame(data=clusterDF['cluster'].value_counts())  # clusterの要素の値と数
    countByCluster.reset_index(inplace=True, drop=False)
    countByCluster.columns = ['cluster', 'clusterCount']

    preds = pd.concat([labelsDF, clusterDF], axis=1)
    preds.columns = ['trueLabel', 'cluster']

    countByLabel = pd.DataFrame(data=preds.groupby('trueLabel').count())

    countMostFreq = pd.DataFrame(data=preds.groupby('cluster').agg(lambda x: x.value_counts().iloc[0]))
    countMostFreq.reset_index(inplace=True, drop=False)
    countMostFreq.columns = ['cluster', 'countMostFrequent']

    accuracyDF = countMostFreq.merge(countByCluster,
                                     left_on="cluster", right_on="cluster")

    overallAccuracy = accuracyDF.countMostFrequent.sum() / accuracyDF.clusterCount.sum()

    accuracyByLabel = accuracyDF.countMostFrequent / accuracyDF.clusterCount

    return countByCluster, countByLabel, countMostFreq, accuracyDF, overallAccuracy, accuracyByLabel
# ------------------------------------------------------------------------------------------------------------------
# HDBSCAN


min_cluster_size = 20
min_samples = 20
alpha = 1.0
cluster_selection_method = 'leaf'

hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                      min_samples=min_samples, alpha=alpha,
                      cluster_selection_method=cluster_selection_method)

X_train_hdbscanClustered = hdb.fit_predict(X_train)
X_train_hdbscanClustered = pd.DataFrame(data=X_train_hdbscanClustered, index=X_train.index,
                                        columns=['cluster'])

countByCluster_hdbscan, countByLabel_hdbscan, \
    countMostFreq_hdbscan, accuracyDF_hdbscan, \
    overallAccuracy_hdbscan, accuracyByLabel_hdbscan = \
    analyzeCluster(X_train_hdbscanClustered, y_train)

print("Overall accuracy from HDBSCAN: ", overallAccuracy_hdbscan)
# 0.3246203751586667

print("Cluster results for HDBSCAN")
print(countByCluster_hdbscan)
print(accuracyByLabel_hdbscan)
x = pd.DataFrame(accuracyByLabel_hdbscan)
file = open('b.csv', 'w')
file.close()
x.to_csv('b.csv')
x = countByCluster_hdbscan
file = open('c.csv', 'w')
file.close()
x.to_csv('c.csv')
"""
   cluster  clusterCount
0       -1         32708
1        7          4070
2        2          3668
3        1          1096
4        4           773
5        0           120
6        6            49
7        3            38
8        5            20
"""
