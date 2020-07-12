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
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)
print(X_train)
# ここからp.40
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018) 


#dataの分布に大きな不均衡があるのでStratifiedKFoldを用いる
"""
概要
分布に大きな不均衡がある場合に用いるKFold．
分布の比率を維持したままデータを訓練用とテスト用に分割する．

オプション(引数)
n_split：データの分割数．つまりk．検定はここで指定した数値の回数おこなわれる．
shuffle：Trueなら連続する数字でグループ分けせず，ランダムにデータを選択する．
random_state：乱数のシードを指定できる．

(kFold)
データをk個に分け，n個を訓練用，k-n個をテスト用として使う．
分けられたn個のデータがテスト用として必ず1回使われるようにn回検定する．
"""
