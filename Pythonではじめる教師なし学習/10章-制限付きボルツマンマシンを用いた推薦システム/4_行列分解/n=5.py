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

#------------------------------------------------------------------------
# 行列分解
# 潜在因子5 (m,n) -> (m,5)*(5,n)

n_latent_factors = 5

user_input = Input(shape=[1], name='user')
user_embedding = Embedding(input_dim=n_users + 1, 
                           output_dim=n_latent_factors, 
                           name='user_embedding')(user_input)
user_vec = Flatten(name='flatten_users')(user_embedding)

movie_input = Input(shape=[1], name='movie')
movie_embedding = Embedding(input_dim=n_movies + 1, 
                            output_dim=n_latent_factors,
                            name='movie_embedding')(movie_input)
movie_vec = Flatten(name='flatten_movies')(movie_embedding)

product = dot([movie_vec, user_vec], axes=1)
model = Model(inputs=[user_input, movie_input], outputs=product)
model.compile('adam', 'mean_squared_error')

history = model.fit(x=[X_train.newUserId, X_train.newMovieId], 
                    y=X_train.rating, epochs=100, 
                    validation_data=([X_valid.newUserId, X_valid.newMovieId], X_valid.rating), 
                    verbose=1)

pd.Series(history.history['val_loss'][10:]).plot(logy=False)
plt.xlabel("Epoch")
plt.ylabel("Validation Error")

file_name = '教師なし教科書/10章-制限付きボルツマンマシンを用いた推薦システム/4_行列分解/result_n=5/'
plt.savefig(file_name + 'figure.png')
with open(file_name + 'result.txt', 'w') as f:
    print(f"Minimum MSE: {round(min(history.history['val_loss']),3)}", file=f)