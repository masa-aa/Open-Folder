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


print(data.describe()) 
"""
data.describe()
count: 要素の個数
unique: ユニークな（一意な）値の要素の個数
top: 最頻値（mode）
freq: 最頻値の頻度（出現回数）
mean: 算術平均
std: 標準偏差
min: 最小値
max: 最大値
50%: 中央値（median）
25%, 75%: 1/4分位数、3/4分位数
"""

print(data.columns) # 列の名前を列挙

print("Number of fraudulent transactions:",data['Class'].sum()) #Classが1のもの(不正であるトランザクションを数える)