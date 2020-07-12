'''Main'''
import numpy as np
import pandas as pd
import os

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

# jupyter notebookで使う
# %matplotlib inline

'''Data Prep'''
from sklearn import preprocessing as pp 
from scipy.stats import pearsonr 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
from sklearn.metrics import precision_recall_curve, average_precision_score 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report 

'''Algos'''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


# Data Preparation

# データの読み込み
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)
#------------------------------------------------------------------------------------------

dataX = data.copy().drop(['Class'],axis=1)
dataY = data['Class'].copy() 
featuresToScale = dataX.drop(['Time'],axis=1).columns
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True) 
dataX.loc[:,featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# ここからp.39
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)

"""

データセットの全てを使って学習（トレーニング）して、同じデータでテストをしては意味がありません。 
テストの答えを元に分類器が学習してしまうので、不用意にスコアが高くなってしまいます。

それを避けるために、 train_test_split 関数を使ってデータを分割しています。 
train_test_split 関数はデータをランダムに、好きの割合で分割できる便利な関数です。

X_train: トレーニング用の特徴行列（「アルコール度数」「密度」「クエン酸」などのデータ）
X_test: テスト用の特徴行列
y_train: トレーニング用の目的変数（「美味しいワイン」か「そうでもないワインか」）
y_test: テスト用の目的変数


train_test_split には以下のような引数を与えます。

第一引数: 特徴行列 X
第二引数: 目的変数 y
test_size=: テスト用のデータを何割の大きさにするか
test_size=0.3 で、3割をテスト用のデータとして置いておけます
random_state=: データを分割する際の乱数のシード値
同じ結果が返るように 0 を指定していますが、これは勉強用であり普段は指定しません
"""