# Import libraries
'''Main and Data Viz and Data Prep and Model Evaluation and Algorithms'''
import csv
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
# 階層クラスタリング


Z = fastcluster.linkage_vector(X_train, method='ward',
                               metric='euclidean')

Z_dataFrame = pd.DataFrame(data=Z, columns=['clusterOne',
                                            'clusterTwo', 'distance', 'newClusterSize'])


print(Z_dataFrame[:20])

print(Z_dataFrame[42521:])

distance_threshold = 100  # 32個のクラスタが得られる
clusters = fcluster(Z, distance_threshold, criterion='distance')
X_train_hierClustered = pd.DataFrame(data=clusters,
                                     index=X_train.index, columns=['cluster'])
print("Number of distinct clusters: ",
      len(X_train_hierClustered['cluster'].unique()))

countByCluster_hierClust, countByLabel_hierClust, \
    countMostFreq_hierClust, accuracyDF_hierClust, \
    overallAccuracy_hierClust, accuracyByLabel_hierClust = \
    analyzeCluster(X_train_hierClustered, y_train)

print("Overall accuracy from hierarchical clustering: ",
      overallAccuracy_hierClust)  # 0.3651685393258427
print("Accuracy by cluster for hierarchical clustering")

print(accuracyByLabel_hierClust)

"""
    clusterOne  clusterTwo      distance  newClusterSize
0      39786.0     39787.0  0.000000e+00             2.0
1      39788.0     42542.0  0.000000e+00             3.0
2      42538.0     42539.0  0.000000e+00             2.0
3      42540.0     42544.0  0.000000e+00             3.0
4      42541.0     42545.0  3.399350e-17             4.0
5      42543.0     42546.0  5.139334e-17             7.0
6      33251.0     33261.0  1.561313e-01             2.0
7      42512.0     42535.0  3.342654e-01             2.0
8      42219.0     42316.0  3.368231e-01             2.0
9       6112.0     21928.0  3.384368e-01             2.0
10     33248.0     33275.0  3.583819e-01             2.0
11     33253.0     33265.0  3.595331e-01             2.0
12     33258.0     42552.0  3.719377e-01             3.0
13     20430.0     23299.0  3.757307e-01             2.0
14      5455.0     32845.0  3.828709e-01             2.0
15     28615.0     30306.0  3.900294e-01             2.0
16      9056.0      9769.0  3.967378e-01             2.0
17     11162.0     13857.0  3.991124e-01             2.0
18     33270.0     42548.0  3.995620e-01             3.0
19     17422.0     17986.0  4.061704e-01             2.0

42521     85038.0     85043.0  132.715723          3969.0
42522     85051.0     85052.0  141.386569          2899.0
42523     85026.0     85027.0  146.976703          2351.0
42524     85048.0     85049.0  152.660192          5691.0
42525     85036.0     85059.0  153.512281          5956.0
42526     85033.0     85044.0  160.825959          2203.0
42527     85055.0     85061.0  163.701428           668.0
42528     85062.0     85066.0  168.199295          6897.0
42529     85054.0     85060.0  168.924039          9414.0
42530     85028.0     85064.0  185.215769          3118.0
42531     85067.0     85071.0  187.832588         15370.0
42532     85056.0     85073.0  203.212147         17995.0
42533     85057.0     85063.0  205.285993          9221.0
42534     85068.0     85072.0  207.902660          5321.0
42535     85069.0     85075.0  236.754581          9889.0
42536     85070.0     85077.0  298.587755         16786.0
42537     85058.0     85078.0  309.946867         16875.0
42538     85074.0     85079.0  375.698458         34870.0
42539     85065.0     85080.0  400.711547         37221.0
42540     85076.0     85081.0  644.047472         42542.0
"""
"""
0     0.304124
1     0.219001
2     0.228311
3     0.379722
4     0.240064
5     0.272011
6     0.314560
7     0.263930
8     0.246138
9     0.318942
10    0.302752
11    0.269772
12    0.335717
13    0.330403
14    0.346320
15    0.440141
16    0.744155
17    0.502227
18    0.294118
19    0.236111
20    0.254727
21    0.241042
22    0.317979
23    0.308771
24    0.284314
25    0.243243
26    0.500000
27    0.289157
28    0.365283
29    0.479693
30    0.393559
31    0.340875
"""
