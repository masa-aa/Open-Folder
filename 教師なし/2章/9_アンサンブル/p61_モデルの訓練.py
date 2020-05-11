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

penalty = 'l2' # 12じゃないよ
C = 1.0
class_weight = 'balanced'
random_state = 2018
solver = 'liblinear'
n_jobs = 1

logReg = LogisticRegression(penalty=penalty, C=C, 
            class_weight=class_weight, random_state=random_state, 
                            solver=solver, n_jobs=n_jobs)
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

params_xGB = {
    'nthread':3, #number of cores
    'learning rate': 0.3, #range 0 to 1, default 0.3
    'gamma': 0, #range 0 to infinity, default 0 
        # increase to reduce complexity (increase bias, reduce variance)
    'max_depth': 6, #range 1 to infinity, default 6
    'min_child_weight': 1, #range 0 to infinity, default 1
    'max_delta_step': 0, #range 0 to infinity, default 0
    'subsample': 1.0, #range 0 to 1, default 1
        # subsample ratio of the training examples
    'colsample_bytree': 1.0, #range 0 to 1, default 1 
        # subsample ratio of features
    'objective':'binary:logistic',
    'num_class':1,
    'eval_metric':'logloss',
    'seed':2018,
    'silent':1
}

params_lightGB = {
    'task': 'train',
    'application':'binary',
    'num_class':1,
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'metric_freq':50,
    'is_training_metric':False,
    'max_depth':4,
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'bagging_seed': 2018,
    'verbose': 0,
    'num_threads':3
}

#ロジスティック
trainingScores = []
cvScores = []
predictionsBasedOnKFoldsLogisticRegression = pd.DataFrame(data=[],index=y_train.index,columns=[0,1]) # 予測
model = logReg

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),y_train.ravel()): # [[1,2,3],[4,5,6],[7,8,9]].ravel()=[1,2,3,4,5,6,7,8,9]
    X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:] # ilocは行番号,列番号で指定する
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

    model.fit(X_train_fold, y_train_fold) # 学習用のデータと結果を学習する
    loglossTraining = log_loss(y_train_fold, model.predict_proba(X_train_fold)[:,1]) # 対数損失の計算
    # predict_proba:[データ数]行 × [次元数]列の特徴量行列 X を引数にして、各データがそれぞれのクラスに所属する確率を返す
    
    trainingScores.append(loglossTraining)

    predictionsBasedOnKFoldsLogisticRegression.loc[X_cv_fold.index,:] = model.predict_proba(X_cv_fold)
    loglossCV = log_loss(y_cv_fold,predictionsBasedOnKFoldsLogisticRegression.loc[X_cv_fold.index,1])

    cvScores.append(loglossCV)
loglossLogisticRegression = log_loss(y_train, predictionsBasedOnKFoldsLogisticRegression.loc[:,1])

# ランダムフォレスト
trainingScores = []
cvScores = []
predictionsBasedOnKFoldsRandomForests = pd.DataFrame(data=[],index=y_train.index,columns=[0,1]) # 予測
model = RFC
for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),y_train.ravel()): # [[1,2,3],[4,5,6],[7,8,9]].ravel()=[1,2,3,4,5,6,7,8,9]
    X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:] # ilocは行番号,列番号で指定する
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

    model.fit(X_train_fold, y_train_fold) # 学習用のデータと結果を学習する
    loglossTraining = log_loss(y_train_fold, model.predict_proba(X_train_fold)[:,1]) # 対数損失の計算
    # predict_proba:[データ数]行 × [次元数]列の特徴量行列 X を引数にして、各データがそれぞれのクラスに所属する確率を返す
    
    trainingScores.append(loglossTraining)

    predictionsBasedOnKFoldsRandomForests.loc[X_cv_fold.index,:] = model.predict_proba(X_cv_fold)
    loglossCV = log_loss(y_cv_fold,predictionsBasedOnKFoldsRandomForests.loc[X_cv_fold.index,1])

    cvScores.append(loglossCV)
loglossRandomForestaClassifier = log_loss(y_train, predictionsBasedOnKFoldsRandomForests.loc[:,1])

# XGBoost
trainingScores = []
cvScores = []
predictionsBasedOnKFoldsXGBoostGradientBoosting = pd.DataFrame(data=[],index=y_train.index,columns=['prediction'])

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),y_train.ravel()):
    X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
    
    dtrain = xgb.DMatrix(data=X_train_fold, label=y_train_fold)
    dCV = xgb.DMatrix(data=X_cv_fold)
    
    bst = xgb.cv(params_xGB, dtrain, num_boost_round=2000, nfold=5, early_stopping_rounds=200, verbose_eval=50)
    # 2000回訓練して過去200回で改善しなければ停止する(過剰適合対策). 50回ごとに表示

    best_rounds = np.argmin(bst['test-logloss-mean']) #loglossが最小値となるindexを返す.
    bst = xgb.train(params_xGB, dtrain, best_rounds) # 一番いいやつで訓練
    
    loglossTraining = log_loss(y_train_fold, bst.predict(dtrain))
    trainingScores.append(loglossTraining)
    
    predictionsBasedOnKFoldsXGBoostGradientBoosting.loc[X_cv_fold.index,'prediction'] = bst.predict(dCV)
    loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFoldsXGBoostGradientBoosting.loc[X_cv_fold.index,'prediction'])
    cvScores.append(loglossCV)
    
loglossXGBoostGradientBoosting = log_loss(y_train, predictionsBasedOnKFoldsXGBoostGradientBoosting.loc[:,'prediction'])

# LightGBM
trainingScores = []
cvScores = []
predictionsBasedOnKFoldsLightGBMGradientBoosting = pd.DataFrame(data=[],index=y_train.index,columns=['prediction'])

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),y_train.ravel()):
    X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
    
    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)
    gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=2000,valid_sets=lgb_eval, early_stopping_rounds=200)
    
    loglossTraining = log_loss(y_train_fold, gbm.predict(X_train_fold, num_iteration=gbm.best_iteration))
    trainingScores.append(loglossTraining)
    
    predictionsBasedOnKFoldsLightGBMGradientBoosting.loc[X_cv_fold.index,'prediction'] = gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration) 
    loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFoldsLightGBMGradientBoosting.loc[X_cv_fold.index,'prediction'])
    cvScores.append(loglossCV)
    
loglossLightGBMGradientBoosting = log_loss(y_train, predictionsBasedOnKFoldsLightGBMGradientBoosting.loc[:,'prediction'])


# アンサンブル

# 訓練セットを加える
predictionsBasedOnKFoldsFourModels = pd.DataFrame(data=[],index=y_train.index)
predictionsBasedOnKFoldsFourModels = predictionsBasedOnKFoldsFourModels.join(
    predictionsBasedOnKFoldsLogisticRegression.loc[:,1].astype(float), \
    how='left').join(predictionsBasedOnKFoldsRandomForests.loc[:,1] \
    .astype(float),how='left',rsuffix="2").join( \
    predictionsBasedOnKFoldsXGBoostGradientBoosting.loc[:,'prediction'].astype(float), \
    how='left',rsuffix="3").join( \
    predictionsBasedOnKFoldsLightGBMGradientBoosting.loc[:,'prediction'].astype(float), \
    how='left',rsuffix="4")
# loc.[:,1]でpredictionのところだけ取ってくる.
predictionsBasedOnKFoldsFourModels.columns = \
    ['predsLR','predsRF','predsXGB','predsLightGBM']

X_trainWithPredictions = X_train.merge(predictionsBasedOnKFoldsFourModels,left_index=True,right_index=True)

#----------------------------------------------------------------------------------------------------------
# モデルの訓練

trainingScores = []
cvScores = []
predictionsBasedOnKFoldsEnsemble = pd.DataFrame(data=[],index=y_train.index,columns=['prediction'])

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):
    X_train_fold, X_cv_fold = X_trainWithPredictions.iloc[train_index,:], X_trainWithPredictions.iloc[cv_index,:]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
    
    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)
    gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=2000,valid_sets=lgb_eval, early_stopping_rounds=200)
    
    loglossTraining = log_loss(y_train_fold, gbm.predict(X_train_fold, num_iteration=gbm.best_iteration))
    trainingScores.append(loglossTraining)
    
    predictionsBasedOnKFoldsEnsemble.loc[X_cv_fold.index,'prediction'] = gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration) 
    loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFoldsEnsemble.loc[X_cv_fold.index,'prediction'])
    cvScores.append(loglossCV)
    
    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)
    
loglossEnsemble = log_loss(y_train, predictionsBasedOnKFoldsEnsemble.loc[:,'prediction'])
print('Ensemble Log Loss: ', loglossEnsemble)