'''Main'''
import numpy as np
import pandas as pd
import os, time, re
import pickle, gzip

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
from sklearn.metrics import roc_curve, auc, roc_auc_score

'''Algos'''
import lightgbm as lgb

'''TensorFlow and Keras'''
import tensorflow as tf
from tensorflow import keras
K = keras.backend

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mse, binary_crossentropy


# To make the output stable across runs
tf.random.set_seed(42)
np.random.seed(42)



# Load the data
current_path = os.getcwd()
file = os.path.sep.join(['', 'datasets', 'credit_card_data', 'credit_card.csv'])
data = pd.read_csv(current_path + file)


dataX = data.copy().drop(['Class', 'Time'], axis=1)
dataY = data['Class'].copy()

# Scale data
featuresToScale = dataX.columns
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

# split
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, 
                                       random_state=2018, stratify=dataY)

# 不正な奴90%消す
toDrop = y_train[y_train == 1].sample(frac=0.90, random_state=2018)  #不正のものを9割消す
X_train.drop(labels=toDrop.index, inplace=True)
y_train.drop(labels=toDrop.index, inplace=True)

# Define evaluation function and plotting function
def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF) - 
                   np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss

number = 1
file_name = '教師なし教科書/9章-半教師あり学習/4_半教師ありモデル/result/'

def plotResults(trueLabels, anomalyScores, returnPreds = False):
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, _ = \
        precision_recall_curve(preds['trueLabel'], 
                               preds['anomalyScore'])
    average_precision = average_precision_score( 
                        preds['trueLabel'], preds['anomalyScore'])
    
    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    plt.title('Precision-Recall curve: Average Precision = {0: 0.2f} '.format(average_precision))

    global number
    plt.savefig(file_name + 'figure_No' + str(number) + '.png')
    number += 1

    fpr, tpr, _ = roc_curve(preds['trueLabel'], 
                                     preds['anomalyScore'])
    areaUnderROC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Area under the \
        curve = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")

    plt.savefig(file_name + 'figure_No' + str(number) + '.png')
    number += 1
    plt.gca().clear()
    
    if returnPreds==True:
        return preds, average_precision

# thresholdに再現率を入れると適合率を返してくれる.  
def precisionAnalysis(df, column, threshold):
    df.sort_values(by=column, ascending=False, inplace=True)
    threshold_value = threshold*df.trueLabel.sum() # 真陽性の数(再現率=真陽性/(真陽性+偽陰性))
    i = 0
    j = 0
    # j:真陽性+偽陽性(陽性の予測) i:真陽性
    while i <= threshold_value:
        if df.iloc[j]["trueLabel"] == 1:
            i += 1
        j += 1
    return df, i / j


#-------------------------------------------------------------------------------------------------------------

oversample_multiplier = 100

X_train_original = X_train.copy()
y_train_original = y_train.copy()
X_test_original = X_test.copy()
y_test_original = y_test.copy()

# オーバーサンプリング 不正をかさ増し
X_train_oversampled = X_train.copy()
y_train_oversampled = y_train.copy()
X_train_oversampled = X_train_oversampled.append( \
        [X_train_oversampled[y_train==1]]*oversample_multiplier, \
        ignore_index=False)
y_train_oversampled = y_train_oversampled.append( \
        [y_train_oversampled[y_train==1]]*oversample_multiplier, \
        ignore_index=False)

X_train = X_train_oversampled.copy()
y_train = y_train_oversampled.copy()

# 過完備線形オートエンコーダ *0.0001の正則化ペナルティ, 2%のドロップアウト 29 -> 40 -> 29
model = Sequential()
model.add(Dense(units=40, activation='linear', activity_regularizer=regularizers.l1(1e-4),
                input_dim=29, name='hidden_layer'))
model.add(Dropout(0.02))
model.add(Dense(units=29, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

num_epochs = 100
batch_size = 64

history = model.fit(x=X_train, y=X_train, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                    validation_split=0.20, verbose=1)

# オートエンコーダで学習した表現と元のデータを用いてLightGBM
layer_name = 'hidden_layer'

# hidden_layerを取得
intermediate_layer_model  = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output_train = intermediate_layer_model.predict(X_train_original)
intermediate_output_test  = intermediate_layer_model.predict(X_test_original)

intermediate_output_trainDF = pd.DataFrame(data=intermediate_output_train,index=X_train_original.index)
intermediate_output_testDF  = pd.DataFrame(data=intermediate_output_test,index=X_test_original.index)

# hidden_layerをマージ
X_train = X_train_original.merge(intermediate_output_trainDF, left_index=True,right_index=True)
X_test  = X_test_original.merge(intermediate_output_testDF, left_index=True,right_index=True)
y_train = y_train_original.copy()


# LightGBM
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

params_lightGB = {
    'task': 'train',
    'application':'binary',
    'num_class':1,
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'metric_freq':50,
    'is_training_metric':False,
    'max_depth':-1,
    'num_leaves': 31,
    'learning_rate': 0.001,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'bagging_seed': 2018,
    'verbose': 0,
    'num_threads':16
}
trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],index=y_train.index,  columns=['prediction'])

with open(file_name + 'result.txt', 'w') as f:
    for train_index, cv_index in k_fold.split(np.zeros(len(X_train)), y_train.ravel()):
        X_train_fold, X_cv_fold = X_train.iloc[train_index,:], X_train.iloc[cv_index,:]
        y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
        
        lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
        lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)
        gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=5000,
                    valid_sets=lgb_eval, early_stopping_rounds=200)
        
        loglossTraining = log_loss(y_train_fold, gbm.predict(X_train_fold,num_iteration=gbm.best_iteration))
        trainingScores.append(loglossTraining)
        
        predictionsBasedOnKFolds.loc[X_cv_fold.index,'prediction'] = gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration) 
        loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFolds.loc[X_cv_fold.index,'prediction'])
        cvScores.append(loglossCV)
        
        print(f'Training Log Loss: {round(loglossTraining,4)}', file=f)
        print(f'CV Log Loss: {round(loglossCV,4)}', file=f)

    

    # result
    loglossLightGBMGradientBoosting = log_loss(y_train, predictionsBasedOnKFolds.loc[:,'prediction'])
    print(f'LightGBM Gradient Boosting Log Loss: {round(loglossLightGBMGradientBoosting,4)}', file=f)

    preds, average_precision = plotResults(y_train, predictionsBasedOnKFolds.loc[:,'prediction'], True)
    
    predictions = pd.Series(data=gbm.predict(X_test, num_iteration=gbm.best_iteration),index=X_test.index)
    preds, average_precision = plotResults(y_test, predictions, True)

    preds, precision = precisionAnalysis(preds, "anomalyScore", 0.75)

    print(f'{round(precision,4)}', file=f)
    
    featuresImportance = pd.DataFrame(data=list(gbm.feature_importance()), index=X_train.columns,columns=['featImportance'])
    featuresImportance = featuresImportance/featuresImportance.sum()
    featuresImportance.sort_values(by='featImportance', ascending=False, inplace=True)
    print("featuresImportance", file=f)
    print(featuresImportance, file=f)

