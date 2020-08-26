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
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error
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
data_train = np.loadtxt(current_path + file +
                        "ECGFiveDays\\ECGFiveDays_TRAIN.tsv")
X_train = to_time_series_dataset(data_train[:, 1:])
y_train = data_train[:, 0].astype(np.int)

data_test = np.loadtxt(current_path + file +
                       "ECGFiveDays\\ECGFiveDays_TEST.tsv")
X_test = to_time_series_dataset(data_test[:, 1:])
y_test = data_test[:, 0].astype(np.int)
file = "教師なし教科書\\13章-時系列クラスタリング\\3_ECGFiveDays_k_shape\\result\\"

# Prepare the data - Scale
X_train = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_train)
X_test = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_test)


# k-Shape Algorithm
# Train using k-Shape
ks = KShape(n_clusters=2, max_iter=100, n_init=100, verbose=0)
ks.fit(X_train)

# Make predictions on train set and calculate adjusted Rand index
preds = ks.predict(X_train)
ars = adjusted_rand_score(data_train[:, 0], preds)
print("Adjusted Rand Index:", ars)

# Make predictions on test set and calculate adjusted Rand index
preds_test = ks.predict(X_test)
ars = adjusted_rand_score(data_test[:, 0], preds_test)
print("Adjusted Rand Index on Test Set:", ars)

# 訓練セットがちいさいから結果が悪い train 23 test 861
# Adjusted Rand Index: 0.668041237113402
# Adjusted Rand Index on Test Set: 0.012338817789874643
