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
file = "教師なし教科書\\13章-時系列クラスタリング\\4_ECG5000_k_shape\\result\\"
# Basic summary statistics
print("Number of time series:", len(data_train))
print("Number of unique classes:", len(np.unique(data_train[:, 0])))
print("Time series length:", len(data_train[0, 1:]))

# Calculate number of readings per class
print("Number of time series in class 1.0:",
      len(data_train[data_train[:, 0] == 1.0]))
print("Number of time series in class 2.0:",
      len(data_train[data_train[:, 0] == 2.0]))
print("Number of time series in class 3.0:",
      len(data_train[data_train[:, 0] == 3.0]))
print("Number of time series in class 4.0:",
      len(data_train[data_train[:, 0] == 4.0]))
print("Number of time series in class 5.0:",
      len(data_train[data_train[:, 0] == 5.0]))


# Display readings from each class
for j in np.unique(data_train[:, 0]):
    dataPlot = data_train[data_train[:, 0] == j]
    cnt = len(dataPlot)
    dataPlot = dataPlot[:, 1:].mean(axis=0)
    print(" Class ", j, " Count ", cnt)
    plt.plot(dataPlot)
    plt.savefig(file + "class_" + str(j) + ".png")
    plt.close()
    plt.clf()

# Prepare the data - Scale
X_train = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_train)
X_test = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_test)
