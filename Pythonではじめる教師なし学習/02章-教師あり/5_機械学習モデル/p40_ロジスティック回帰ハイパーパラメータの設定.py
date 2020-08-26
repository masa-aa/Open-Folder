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

penalty = 'l2' # 12じゃないよ
C = 1.0
class_weight = 'balanced'
random_state = 2018
solver = 'liblinear'
n_jobs = 1

logReg = LogisticRegression(penalty=penalty, C=C, 
            class_weight=class_weight, random_state=random_state, 
                            solver=solver, n_jobs=n_jobs)
# LogisticRegression:ロジスティクス回帰をしてくれる優れもの.
"""
引数

penalty:
    文字列を指定．正則化をL1ノルムでやるかL2ノルムでやるかを，l1 または l2 から選択．
    L2はL1と比較すると外れ値に対して過敏ではなく,ほとんどすべての特徴量に0でない重みを
    与えるので解が安定する. L1では重要な特徴量に大きな重みを与え, 他の特徴量には0に近い重みしか与えない.

C(>0):
    数値を指定．デフォルトの値は1.0．これは単に正則化項の係数．小さいほど正則化が強くなる.
    今回は284807中492しかないので小さくしない.

class_weight:
    ディクショナリまたは balanced を指定．クラスに対する重みをディクショナリで指定できる．
    指定しない場合は全てのクラスに1が設定されている．balanced を指定すると，
    y の値により n_samples / (n_classes * np.bincount(y)) を計算することで自動的に重みを調整する.
    balanced にすることにより, 事例数の少ないクラスに大きい重みを与えている.

random_state:
    乱数のタネの指定．再現性より2018に固定.

solver:
    文字列を指定．最適解の探索手法を newton-cg，lbfgs，liblinear，sag から選択する．

n_jobs:
    整数を指定．デフォルトは1．フィットおよび予測の際に用いるスレッドの数を指定．
    -1 を指定した場合は計算機に載っている全スレッド分確保される．
"""

# https://data-science.gr.jp/implementation/iml_sklearn_logistic_regression.html