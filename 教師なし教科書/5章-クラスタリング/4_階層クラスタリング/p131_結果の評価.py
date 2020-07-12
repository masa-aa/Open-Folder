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
# Hierarchical clustering
import fastcluster
from scipy.cluster.hierarchy import dendrogram, cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

cutoff = 99
Z = fastcluster.linkage_vector(X_train_PCA.loc[:,0:cutoff], \
                               method='ward', metric='euclidean')
Z_dataFrame = pd.DataFrame(data=Z, \
    columns=['clusterOne','clusterTwo','distance','newClusterSize'])

distance_threshold = 20
#与えられたlinkage matrixで定義された階層クラスタリングから，フラットなクラスタを形成します．
clusters = fcluster(Z, distance_threshold, criterion='distance') 

X_train_hierClustered = \
    pd.DataFrame(data=clusters,index=X_train_PCA.index,columns=['cluster'])
print("Number of distinct clusters: ", \
      len(X_train_hierClustered['cluster'].unique())) # 20個のクラスタが得られる.

countByCluster_hierClust, countByLabel_hierClust, \
    countMostFreq_hierClust, accuracyDF_hierClust, \
    overallAccuracy_hierClust, accuracyByLabel_hierClust \
    = analyzeCluster(X_train_hierClustered, y_train)

print("Overall accuracy from hierarchical clustering: ", \
      overallAccuracy_hierClust) # 精度が77.066%

print("Accuracy by cluster for hierarchical clustering") # 階層型クラスタリングのクラスタ別精度
print(accuracyByLabel_hierClust)
"""
0     0.657038
1     0.682914
2     0.570198
3     0.559322
4     0.969185
5     0.991484
6     0.984788
7     0.982719
8     0.985955
9     0.989979
10    0.991837
11    0.420673
12    0.620503
13    0.479452
14    0.960112
15    0.743284
16    0.948679
17    0.950038
18    0.989378
19    0.978018
"""