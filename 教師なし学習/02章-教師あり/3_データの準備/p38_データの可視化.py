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


count_classes = pd.value_counts(data['Class'],sort=True).sort_index() 
# value_counts(data['Class'],sort=True) ユニークな要素の値がindex,その出現個数がdataとなる.出現回数が多いものから順にソートされる
# sort_index() 行名に従って行方向にソートする
print(count_classes)
ax = sns.barplot(x=count_classes.index,y=tuple(count_classes/len(data))) # x軸がclassの値, y軸が0 or 1の出現率
ax.set_title("Frequency Percentage by Class") #タイトル
ax.set_xlabel("Class") # x軸のラベル
ax.set_ylabel("Frequency Percentage") # y軸のラベル

mpl.pyplot.show()