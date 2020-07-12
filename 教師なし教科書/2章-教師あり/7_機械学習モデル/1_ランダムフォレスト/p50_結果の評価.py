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

# 標準化
dataX = data.copy().drop(['Class'],axis=1)
dataY = data['Class'].copy() 
featuresToScale = dataX.drop(['Time'],axis=1).columns
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True) 
dataX.loc[:,featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# 訓練セットの作成
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)

# k分割交差検証セットの作成
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018) 

# ハイパーパラメータの設定
n_estimators = 10 # 決定木の数
max_features = 'auto'
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
min_weight_fraction_leaf = 0.0
max_leaf_nodes = None
bootstrap = True
oob_score = False
n_jobs = -1
random_state = 2018
class_weight = 'balanced'

RFC = RandomForestClassifier(n_estimators=n_estimators, 
        max_features=max_features, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf, 
        max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, 
        oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, 
        class_weight=class_weight)

# モデルの訓練

trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],index=y_train.index,columns=[0,1]) # 予測(0に入る確率と1に入る確率)
model = RFC
# ここからはロジスティックと同じ
for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),y_train.ravel()): # [[1,2,3],[4,5,6],[7,8,9]].ravel()=[1,2,3,4,5,6,7,8,9]
    X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:] # ilocは行番号,列番号で指定する
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

    model.fit(X_train_fold, y_train_fold) # 学習用のデータと結果を学習する
    loglossTraining = log_loss(y_train_fold, model.predict_proba(X_train_fold)[:,1]) # 対数損失の計算
    # predict_proba:[データ数]行 × [次元数]列の特徴量行列 X を引数にして、各データがそれぞれのクラスに所属する確率を返す
    
    trainingScores.append(loglossTraining)

    predictionsBasedOnKFolds.loc[X_cv_fold.index,:] = model.predict_proba(X_cv_fold)
    loglossCV = log_loss(y_cv_fold,predictionsBasedOnKFolds.loc[X_cv_fold.index,1])

    cvScores.append(loglossCV)

    print("Training Log Loss: ",loglossTraining)
    print("CV Log Loss: ", loglossCV)

loglossLogisticRegression = log_loss(y_train, predictionsBasedOnKFolds.loc[:,1])

print("Logistic Regression Log Loss: ", loglossLogisticRegression)

#------------------------------------------------------------------------------------------
# 結果の評価

# 平均適合率
preds = pd.concat([y_train,predictionsBasedOnKFolds.loc[:,1]], axis=1) # データを横方向に連結
preds.columns = ['trueLabel','prediction']
predictionsBasedOnKFoldsLogisticRegression = preds.copy()
precision, recall, thresholds = precision_recall_curve(preds['trueLabel'],preds['prediction']) # ある閾値の時の適合率、再現率の値を取得
# 日本語訳　精度、再現率、閾値
average_precision = average_precision_score(preds['trueLabel'],preds['prediction']) 
# 予測スコアから平均適合率を計算する. 適合率-再現率スコアの下の部分に対応.
plt.step(recall, precision, color='k', alpha=0.7, where='post') #階段plotを表示. x座標がrecall,y座標がprecision
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))

# mpl.pyplot.show() # 平均適合率が79%であることがわかる.



# acROCの計算(受信者動作特性曲線下の面積) 偽陽性を可能な限り低く保ちながら不正をどれだけ捉えれるか.
fpr, tpr, thresholds = roc_curve(preds['trueLabel'],preds['prediction'])
areaUnderROC = auc(fpr, tpr) # auROCを算出

plt.figure()
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: Area under the curve = {0:0.2f}'.format(areaUnderROC))
plt.legend(loc="lower right")
plt.show()

"""
訓練対数損失が交差検証対数損失よりもはるかに小さい -> 過剰適合
過剰適合しているにもかかわらずランダムフォレストの対数損失が
ロジスティック回帰の対数損失の1/10程度なので, いい感じ

適合率-再現率曲線を見ると,80%の適合率を保ちながら, 80%の不正を検出している.(ロジスティックは70%くらいしか検出できない)
平均適合率もロジスティックより高いがauROCより低い
"""