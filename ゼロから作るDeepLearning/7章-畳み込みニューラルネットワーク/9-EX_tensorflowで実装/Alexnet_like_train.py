import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../..")
# # load data

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from IPython.core.display import display
from time import time


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# build model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Reshape, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
Dim1 = [1 << i for i in range(3, 7)]
Dim2 = [1 << i for i in range(3, 7)]
Dim3 = [1 << i for i in range(3, 7)]
Dim4 = [1 << i for i in range(3, 7)]
Dim5 = [1 << i for i in range(3, 7)]

def my_net(dim1, dim2, dim3, dim4, dim5):
    model = Sequential()
    model.add(Input(shape=(28, 28)))
    model.add(Reshape((28, 28, 1)))
    # 1
    model.add(Conv2D(dim1, 3, activation='relu', bias_initializer='ones'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(BatchNormalization())

    # 2
    model.add(Conv2D(dim2, 3, activation='relu', bias_initializer='ones'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 3-5
    model.add(Conv2D(dim3, 3, bias_initializer='zeros'))
    model.add(Conv2D(dim4, 3, bias_initializer='ones'))
    model.add(Conv2D(dim5, 3, bias_initializer='ones'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())
    
    # 全結合層
    model.add(Flatten())
    model.add(Dense(784))
    model.add(Dropout(0.5))
    model.add(Dense(784))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model
start = time()
# print(model.summary())
f_model = './model'
best = []
for dim1 in Dim1:
    for dim2 in Dim2:
        for dim3 in Dim3:
            for dim4 in Dim4:
                for dim5 in Dim5:
                    model = my_net(dim1, dim2, dim3, dim4, dim5)
                    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
                    # cp_cb = ModelCheckpoint(filepath = os.path.join(f_model,'cnn_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
                    history = model.fit(x=x_train, y=y_train, epochs=15, batch_size=128, shuffle=True, validation_split=0.25, callbacks=[early_stopping])
                    best.append((max(history.history["val_acc"]), dim1, dim2, dim3, dim4, dim5))

best.sort(reverse=True)
for i in range(5):
    print("No." + str(i + 1), best[i])
print(time() - start, "sec")
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# print('save the architecture of a model')
# json_string = model.to_json()
# open(os.path.join(f_model,'cnn_model.json'), 'w').write(json_string)
# yaml_string = model.to_yaml()
# open(os.path.join(f_model,'cnn_model.yaml'), 'w').write(yaml_string)
# print('save weights')
# model.save_weights(os.path.join(f_model,'cnn_model_weights.hdf5'))