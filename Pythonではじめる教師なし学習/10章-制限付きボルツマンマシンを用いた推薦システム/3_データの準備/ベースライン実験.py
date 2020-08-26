'''Main'''
import numpy as np
import pandas as pd
import os, time, re
import pickle, gzip, datetime
from datetime import datetime

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

# %matplotlib inline

'''Data Prep and Model Evaluation'''
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error

'''Algos'''
import lightgbm as lgb

'''TensorFlow and Keras'''
import tensorflow as tf
from tensorflow import keras
K = keras.backend

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Lambda
from tensorflow.keras.layers import Embedding, Flatten, dot
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mse, binary_crossentropy


import sys, sklearn

# To make the output stable across runs
tf.random.set_seed(42)
np.random.seed(42)

current_path = os.getcwd()
pickle_file = os.path.sep.join(['', 'datasets', 'movielens_data', 'ratingPickle'])
ratingDF = pd.read_pickle(current_path + pickle_file)

pickle_file = os.path.sep.join(['', 'datasets', 'movielens_data', 'ratingReducedPickle'])
ratingDFX3 = pd.read_pickle(current_path + pickle_file)


# 統計量を計算
n_users = ratingDFX3.userId.unique().shape[0]
n_movies = ratingDFX3.movieId.unique().shape[0]
n_ratings = len(ratingDFX3)
avg_ratings_per_user = n_ratings/n_users


# それぞれ全体の5%を検証セットとテストセットとする.
X_train, X_test = train_test_split(ratingDFX3, test_size=0.10, shuffle=True, random_state=2018)
X_valid, X_test = train_test_split(X_test,     test_size=0.50, shuffle=True, random_state=2018)

# 平均二乗誤差 MSE
# (ユーザ数)*(映画の数)の行列
ratings_train = np.zeros((n_users, n_movies))
ratings_valid = np.zeros((n_users, n_movies))
ratings_test = np.zeros((n_users, n_movies))
for (X,ratings) in [(X_train,ratings_train),(X_valid,ratings_valid),(X_test,ratings_test)]:
    for row in X.itertuples():
        # rating[newUserId, newMovieId] = rating
        ratings[row[6] - 1, row[5] - 1] = row[3]

# flatten
actual_valid = ratings_valid[ratings_valid.nonzero()].flatten()

# ベースライン実験

# experiment 1
# 格付けの平均値である3.5を予測値とした場合
pred_valid = np.zeros((len(X_valid), 1))
# 0のもの (すべて)を3.5にする
pred_valid[pred_valid == 0] = 3.5

naive_prediction = mean_squared_error(pred_valid, actual_valid)
print(f'Mean squared error using naive prediction: {round(naive_prediction,2)}')
# 1.055

# experiment 2
# そのユーザーによる格付けの平均値を他のすべての映画の予測値として用いた場合

ratings_valid_pred = np.zeros((n_users, n_movies))
i = 0
for row in ratings_train:
    # 各々平均を埋め込む
    ratings_valid_pred[i][ratings_valid_pred[i]==0] = np.mean(row[row>0])
    i += 1

pred_valid = ratings_valid_pred[ratings_valid.nonzero()].flatten()
user_average = mean_squared_error(pred_valid, actual_valid)
print(f'Mean squared error using user average: {round(user_average,3)}')
# 0.909

# exiperiment 3
# 対象の映画に対するほかのユーザーの格付けの平均値を予測値とする.
ratings_valid_pred = np.zeros((n_users, n_movies)).T
i = 0
for row in ratings_train.T:
    ratings_valid_pred[i][ratings_valid_pred[i] == 0] = np.mean(row[row > 0])
    i += 1

ratings_valid_pred = ratings_valid_pred.T
pred_valid = ratings_valid_pred[ratings_valid.nonzero()].flatten()
movie_average = mean_squared_error(pred_valid, actual_valid)
print(f'Mean squared error using movie average: {round(movie_average,3)}')
# 0.914