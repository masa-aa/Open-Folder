'''Main'''
import numpy as np
import pandas as pd
import os, time, re
import pickle, gzip, datetime
from os import listdir, walk
from os.path import isfile, join

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Grid

'''Data Prep and Model Evaluation'''
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import adjusted_rand_score
import random

'''Algos'''
import tslearn
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape, TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
import hdbscan


# Load the datasets
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'UCRArchive_2018', ''])
data_train = np.loadtxt(current_path + file + "ECG5000\\ECG5000_TRAIN.tsv")
data_test = np.loadtxt(current_path + file + "ECG5000\\ECG5000_TEST.tsv")

# trainとtestを80%, 20%に分けなおす
data_joined = np.concatenate((data_train, data_test), axis=0)
data_train, data_test = train_test_split(data_joined, test_size=0.20, random_state=2019)

X_train = to_time_series_dataset(data_train[:, 1:])
y_train = data_train[:, 0].astype(np.int)
X_test = to_time_series_dataset(data_test[:, 1:])
y_test = data_test[:, 0].astype(np.int)
file_name = "教師なし教科書\\13章-時系列クラスタリング\\4_ECG5000_k_shape\\result\\"


# Prepare the data - Scale
X_train = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_train)
X_test = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_test)


# Train using k-Shape
ks = KShape(n_clusters=5, max_iter=100, n_init=10, verbose=1, random_state=2019)
ks.fit(X_train)
with open(file_name + 'result.txt', 'w') as f:
    # Predict on train set and calculate adjusted Rand index
    preds = ks.predict(X_train)
    ars = adjusted_rand_score(data_train[:, 0], preds)
    print("Adjusted Rand Index on Training Set:", ars, file=f)

    preds_test = ks.predict(X_test)
    ars = adjusted_rand_score(data_test[:, 0], preds_test)
    print("Adjusted Rand Index on Test Set:", ars, file=f)

    # Evaluate goodness of the clusters
    preds_test = preds_test.reshape(1000, 1)
    preds_test = np.hstack((preds_test, data_test[:, 0].reshape(1000, 1)))
    preds_test = pd.DataFrame(data=preds_test)
    preds_test = preds_test.rename(columns={0: 'prediction', 1: 'actual'})

    counter = 0
    for i in np.sort(preds_test.prediction.unique()):
        print("Predicted Cluster ", i, file=f)
        print(preds_test.actual[preds_test.prediction == i].value_counts(), file=f)
        print("", file=f)
        cnt = preds_test.actual[preds_test.prediction == i].value_counts().iloc[1:].sum()
        counter = counter + cnt
    print("Count of Non-Primary Points: ", counter, file=f)
