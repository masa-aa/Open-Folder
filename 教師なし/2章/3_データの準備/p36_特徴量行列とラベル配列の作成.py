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

# 特徴量行列Xとラベル配列Yの作成
dataX = data.copy().drop(['Class'],axis=1) # drop(['Class'],axis=1) Classの列を消す
dataY = data['Class'].copy() 
# pandasのcopyの挙動はここを見てくれ　https://qiita.com/analytics-hiro/items/e8fdf8652869817851cd

# 特徴量行列Xの標準化
featuresToScale = dataX.drop(['Time'],axis=1).columns # columns:列の名前を列挙
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True) 
# StandardScaler:標準化関数 
"""
copy:
Falseの場合、transformやfit_transformメソッドで変換時に、
変換元のデータを破壊的に変換する

with_mean:
Trueの場合、平均値を0とする。
Falseの場合、y=x/(標準偏差)になる。

with_std
Trueの場合、分散を0とする。
Falseの場合、y=x-(平均値)になる。
"""
dataX.loc[:,featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# 標準化された特徴量を表示
print(dataX.describe())
"""
loc[:,...]
:で列ラベルを指定
featureaToScaleのデータを参照している.

fit_transform(X):配列Xの平均と分散を計算して,記憶し,配列Xに変換を施して,変換後の配列を返す。
"""