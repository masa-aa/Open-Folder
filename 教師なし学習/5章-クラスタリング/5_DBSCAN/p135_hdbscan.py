# Import libraries
'''Main'''
import numpy as np
import pandas as pd
import os, time
import pickle, gzip

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

# %matplotlib inline

'''Data Prep and Model Evaluation'''
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

'''Algorithms'''
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import fastcluster
from scipy.cluster.hierarchy import dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist

# Load the datasets
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'mnist_data', 'mnist.pkl.gz'])

f = gzip.open(current_path+file, 'rb')
train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
f.close()

X_train, y_train = train_set[0], train_set[1]
X_validation, y_validation = validation_set[0], validation_set[1]
X_test, y_test = test_set[0], test_set[1]

# Create Pandas DataFrames from the datasets
train_index = range(0,len(X_train))
validation_index = range(len(X_train), \
                         len(X_train)+len(X_validation))
test_index = range(len(X_train)+len(X_validation), \
                   len(X_train)+len(X_validation)+len(X_test))

X_train = pd.DataFrame(data=X_train,index=train_index)
y_train = pd.Series(data=y_train,index=train_index)

X_validation = pd.DataFrame(data=X_validation,index=validation_index)
y_validation = pd.Series(data=y_validation,index=validation_index)

X_test = pd.DataFrame(data=X_test,index=test_index)
y_test = pd.Series(data=y_test,index=test_index)

# PCAで次元削減(前処理)
from sklearn.decomposition import PCA

n_components = 784
whiten = False
random_state = 2018

pca = PCA(n_components=n_components, whiten=whiten, \
          random_state=random_state)

X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)

#----------------------------------------------------------------------

# クラスタリングの結果の評価
def analyzeCluster(clusterDF, labelsDF):
    # 各クラスタの観測点の数を数えてcountByClusterに格納
    countByCluster = \
        pd.DataFrame(data=clusterDF['cluster'].value_counts()) #clusteの要素の値と数
    countByCluster.reset_index(inplace=True,drop=False) # indexを振りなおす
    countByCluster.columns = ['cluster','clusterCount']
    
    # clusterDFを実際のラベル配列labelsDFと連結
    preds = pd.concat([labelsDF,clusterDF], axis=1) # 横に連結
    preds.columns = ['trueLabel','cluster']
    # 訓練セット内の実際のラベルに対しても観測点の数を数える.(これは変化しない)
    countByLabel = pd.DataFrame(data=preds.groupby('trueLabel').count()) # trueLabel(ラベル配列)のcount

    # 最も多かったラベルをカウント  
    countMostFreq = \
        pd.DataFrame(data=preds.groupby('cluster').agg(lambda x:x.value_counts().iloc[0]))
    countMostFreq.reset_index(inplace=True,drop=False)
    countMostFreq.columns = ['cluster','countMostFrequent']
    # クラスタリングの精度:=(それぞれのクラスタに最も多く存在するラベルの観測値の数の和/訓練セット全体の観測値の数)
    accuracyDF = countMostFreq.merge(countByCluster, \
                        left_on="cluster",right_on="cluster") #左が"cluster",右が"cluster"で結合
    overallAccuracy = accuracyDF.countMostFrequent.sum()/ accuracyDF.clusterCount.sum()
    # 個々のクラスタごとの精度
    accuracyByLabel = accuracyDF.countMostFrequent/ accuracyDF.clusterCount
    
    return countByCluster, countByLabel, countMostFreq, accuracyDF, overallAccuracy, accuracyByLabel
#-----------------------------------------------------------------------------------------------------
# HDBSCAN
import hdbscan

min_cluster_size = 30
min_samples = None
alpha = 1.0
cluster_selection_method = 'eom'

hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, \
        min_samples=min_samples, alpha=alpha, \
        cluster_selection_method=cluster_selection_method)

cutoff = 10
X_train_PCA_hdbscanClustered = \
    hdb.fit_predict(X_train_PCA.loc[:,0:cutoff])

X_train_PCA_hdbscanClustered = \
    pd.DataFrame(data=X_train_PCA_hdbscanClustered, \
    index=X_train.index, columns=['cluster'])

countByCluster_hdbscan, countByLabel_hdbscan, \
    countMostFreq_hdbscan, accuracyDF_hdbscan, \
    overallAccuracy_hdbscan, accuracyByLabel_hdbscan \
    = analyzeCluster(X_train_PCA_hdbscanClustered, y_train)

print("Overall accuracy from HDBSCAN: ",overallAccuracy_hdbscan) # 24.358%

print("Cluster results for HDBSCAN")
print(countByCluster_hdbscan)
"""
   cluster  clusterCount
0       -1         42741
1        5          5139
2        6           943
3        0           480
4        7           294
5        3           252
6        1            72
7        4            47
8        2            32
"""