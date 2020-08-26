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

# Experiment to compare time series clustering algorithms
# Load the datasets
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'UCRArchive_2018', ''])

mypath = current_path + file
d = []
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    for i in dirnames:
        newpath = os.path.sep.join([mypath, i, ""])
        onlyfiles = [f for f in listdir(newpath) if isfile(join(newpath, f))]
        f.extend(onlyfiles)
    d.extend(dirnames)
    break


file_name = "教師なし教科書\\13章-時系列クラスタリング\\7_時系列クラスタリングアルゴリズムの比較\\result\\"


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))
        return (time.time() - self.start_time)


sub_file = "k_shape"
with open(file_name + sub_file + '.txt', 'w') as f:
    # k-Shape Experiment - FULL RUN
    # Create dataframe
    kShapeDF = pd.DataFrame(data=[], index=[v for v in d], columns=["Train ARS", "Test ARS"])
    # Train and Evaluate k-Shape
    timer = ElapsedTimer()
    cnt = 0
    for i in d:
        newpath = os.path.sep.join([mypath, i, ""])

        # Missing_value_and_variable_length_datasets_adjustedは修正データが入ったフォルダ
        if newpath.endswith("Missing_value_and_variable_length_datasets_adjusted\\"):
            continue

        cnt += 1
        print("Dataset ", cnt, file=f)
        # 作者コードだとフォルダ名がReadme.mdより大きいと壊れる.
        _file_name = newpath.split("\\")[-2]
        j = _file_name + "_TRAIN.tsv"
        k = _file_name + "_TEST.tsv"

        data_train = np.loadtxt(newpath + j)
        data_test = np.loadtxt(newpath + k)

        data_joined = np.concatenate((data_train, data_test), axis=0)

        # NaNがあるデータは無視したよ. 使いたいときはMissing_value_and_variable_length_datasets_adjustedにあるデータで置き換えるとよい
        if np.isnan(data_joined).sum():
            print("This dataset contains 'NaN'.", file=f)
            print(file=f)
            continue

        data_train, data_test = train_test_split(data_joined,
                                                 test_size=0.20, random_state=2019)

        X_train = to_time_series_dataset(data_train[:, 1:])
        y_train = data_train[:, 0].astype(np.int)
        X_test = to_time_series_dataset(data_test[:, 1:])
        y_test = data_test[:, 0].astype(np.int)

        X_train = TimeSeriesScalerMeanVariance(mu=0., std=1.) \
            .fit_transform(X_train)
        X_test = TimeSeriesScalerMeanVariance(mu=0., std=1.) \
            .fit_transform(X_test)

        classes = len(np.unique(data_train[:, 0]))
        ks = KShape(n_clusters=classes, max_iter=10, n_init=3, verbose=0)
        ks.fit(X_train)

        print(i, file=f)
        preds = ks.predict(X_train)
        ars = adjusted_rand_score(data_train[:, 0], preds)
        print("Adjusted Rand Index on Training Set:", ars, file=f)
        kShapeDF.loc[i, "Train ARS"] = ars

        preds_test = ks.predict(X_test)
        ars = adjusted_rand_score(data_test[:, 0], preds_test)
        print("Adjusted Rand Index on Test Set:", ars, file=f)
        kShapeDF.loc[i, "Test ARS"] = ars
        print(file=f)

    kShapeTime = timer.elapsed_time()
    print("Time to Run k-Shape Experiment in Minutes:", kShapeTime / 60, file=f)
    kShapeDF.to_pickle(os.path.sep.join([current_path, 'datasets', 'pickled_data', "kShapeDF.pickle"]))


sub_file = "k_means"
with open(file_name + sub_file + '.txt', 'w') as f:
    # k-Means Experiment - FULL RUN
    # Create dataframe
    kMeansDF = pd.DataFrame(data=[], index=[v for v in d],
                            columns=["Train ARS", "Test ARS"])
    timer = ElapsedTimer()
    cnt = 0
    for i in d:
        newpath = os.path.sep.join([mypath, i, ""])

        if newpath.endswith("Missing_value_and_variable_length_datasets_adjusted\\"):
            continue
        cnt += 1
        print("Dataset ", cnt, file=f)

        _file_name = newpath.split("\\")[-2]
        j = _file_name + "_TRAIN.tsv"
        k = _file_name + "_TEST.tsv"

        data_train = np.loadtxt(newpath + j)
        data_test = np.loadtxt(newpath + k)

        data_joined = np.concatenate((data_train, data_test), axis=0)
        if np.isnan(data_joined).sum():
            print("This dataset contains 'NaN'.", file=f)
            print(file=f)
            continue
        data_train, data_test = train_test_split(data_joined, test_size=0.20, random_state=2019)

        X_train = to_time_series_dataset(data_train[:, 1:])
        y_train = data_train[:, 0].astype(np.int)
        X_test = to_time_series_dataset(data_test[:, 1:])
        y_test = data_test[:, 0].astype(np.int)

        X_train = TimeSeriesScalerMeanVariance(mu=0., std=1.) \
            .fit_transform(X_train)
        X_test = TimeSeriesScalerMeanVariance(mu=0., std=1.) \
            .fit_transform(X_test)
        classes = len(np.unique(data_train[:, 0]))
        km = TimeSeriesKMeans(n_clusters=5, max_iter=10, n_init=10,
                              metric="euclidean", verbose=0, random_state=2019)
        km.fit(X_train)

        print(i, file=f)
        preds = km.predict(X_train)
        ars = adjusted_rand_score(data_train[:, 0], preds)
        print("Adjusted Rand Index on Training Set:", ars, file=f)
        kMeansDF.loc[i, "Train ARS"] = ars

        preds_test = km.predict(X_test)
        ars = adjusted_rand_score(data_test[:, 0], preds_test)
        print("Adjusted Rand Index on Test Set:", ars, file=f)
        kMeansDF.loc[i, "Test ARS"] = ars
        print(file=f)
    kMeansTime = timer.elapsed_time()

    print("Time to Run k-Means Experiment in Minutes:", kMeansTime / 60, file=f)
    kMeansDF.to_pickle(os.path.sep.join([current_path, "datasets", "pickled_data",
                                         "kMeansDF.pickle"]))


sub_file = "HDBSCAN"
with open(file_name + sub_file + '.txt', 'w') as f:
    # HDBSCAN Experiment - FULL RUN
    # Create dataframe
    hdbscanDF = pd.DataFrame(data=[], index=[v for v in d],
                             columns=["Train ARS", "Test ARS"])
    timer = ElapsedTimer()
    cnt = 0
    for i in d:
        newpath = os.path.sep.join([mypath, i, ""])

        if newpath.endswith("Missing_value_and_variable_length_datasets_adjusted\\"):
            continue
        cnt += 1
        print("Dataset ", cnt, file=f)

        _file_name = newpath.split("\\")[-2]
        j = _file_name + "_TRAIN.tsv"
        k = _file_name + "_TEST.tsv"

        data_train = np.loadtxt(newpath + j)
        data_test = np.loadtxt(newpath + k)

        data_joined = np.concatenate((data_train, data_test), axis=0)
        if np.isnan(data_joined).sum():
            print("This dataset contains 'NaN'.", file=f)
            print(file=f)
            continue
        data_train, data_test = train_test_split(data_joined, test_size=0.20, random_state=2019)

        X_train = data_train[:, 1:]
        y_train = data_train[:, 0].astype(np.int)
        X_test = data_test[:, 1:]
        y_test = data_test[:, 0].astype(np.int)

        X_train = TimeSeriesScalerMeanVariance(mu=0., std=1.) \
            .fit_transform(X_train)
        X_test = TimeSeriesScalerMeanVariance(mu=0., std=1.)  \
            .fit_transform(X_test)

        classes = len(np.unique(data_train[:, 0]))
        min_cluster_size = 5
        min_samples = None
        alpha = 1.0
        cluster_selection_method = 'eom'
        prediction_data = True

        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              min_samples=min_samples, alpha=alpha,
                              cluster_selection_method=cluster_selection_method,
                              prediction_data=prediction_data)

        print(i, file=f)
        preds = hdb.fit_predict(X_train.reshape(X_train.shape[0],
                                                X_train.shape[1]))
        ars = adjusted_rand_score(data_train[:, 0], preds)
        print("Adjusted Rand Index on Training Set:", ars, file=f)
        hdbscanDF.loc[i, "Train ARS"] = ars

        preds_test = hdbscan.prediction.approximate_predict(hdb,
                                                            X_test.reshape(X_test.shape[0],
                                                                           X_test.shape[1]))
        ars = adjusted_rand_score(data_test[:, 0], preds_test[0])
        print("Adjusted Rand Index on Test Set:", ars, file=f)
        hdbscanDF.loc[i, "Test ARS"] = ars

    hdbscanTime = timer.elapsed_time()
    print("Time to Run HDBSCAN Experiment in Minutes:", hdbscanTime / 60, file=f)

    hdbscanDF.to_pickle(os.path.sep.join([current_path, "datasets", "pickled_data",
                                          "hdbscanDF.pickle"]))


# Compare All Three Experiments
sub_file = "result"
with open(file_name + sub_file + '.txt', 'w') as f:
    print("k-Shape Results", file=f)
    print(kShapeDF.mean(), file=f)
    print(file=f)

    print("k-Means Results", file=f)
    print(kMeansDF.mean(), file=f)
    print(file=f)

    print("HDBSCAN Results", file=f)
    print(hdbscanDF.mean(), file=f)
    print(file=f)

    # Count top place finishes
    timeSeriesClusteringDF = pd.DataFrame(data=[], index=kShapeDF.index,
                                          columns=["kShapeTest",
                                                   "kMeansTest",
                                                   "hdbscanTest"])

    timeSeriesClusteringDF.kShapeTest = kShapeDF["Test ARS"]
    timeSeriesClusteringDF.kMeansTest = kMeansDF["Test ARS"]
    timeSeriesClusteringDF.hdbscanTest = hdbscanDF["Test ARS"]

    tscResults = timeSeriesClusteringDF.copy()

    for i in range(0, len(tscResults)):
        maxValue = tscResults.iloc[i].max()
        tscResults.iloc[i][tscResults.iloc[i] == maxValue] = 1
        minValue = tscResults .iloc[i].min()
        tscResults.iloc[i][tscResults.iloc[i] == minValue] = -1
        medianValue = tscResults.iloc[i].median()
        tscResults.iloc[i][tscResults.iloc[i] == medianValue] = 0

    # Show results
    tscResultsDF = pd.DataFrame(data=np.zeros((3, 3)),
                                index=["firstPlace", "secondPlace", "thirdPlace"],
                                columns=["kShape", "kMeans", "hdbscan"])
    tscResultsDF.loc["firstPlace", :] = tscResults[tscResults == 1].count().values
    tscResultsDF.loc["secondPlace", :] = tscResults[tscResults == 0].count().values
    tscResultsDF.loc["thirdPlace", :] = tscResults[tscResults == -1].count().values
    print(tscResultsDF, file=f)
