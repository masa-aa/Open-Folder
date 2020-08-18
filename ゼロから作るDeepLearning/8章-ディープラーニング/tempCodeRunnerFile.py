# ディレクトリ移動
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../..")

# load data
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from IPython.core.display import display

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# build model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Reshape, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization


def DeepNeuralNetwork():
    """
    input ->
    Conv -> ReLU -> Conv -> ReLU -> Pool ->
    Conv -> ReLU -> Conv -> ReLU -> Pool ->
    Conv -> ReLU -> Conv -> ReLU -> Pool ->
    Affine -> ReLU -> Dropout ->
    Affine -> Dropout -> Softmax ->
    output
    """
    model = Sequential()
    model.add(Input(shape=(28, 28)))
    model.add(Reshape((28, 28, 1)))

    # 1
    model.add(Conv2D(16, 3, activation='relu', bias_initializer='he_normal', padding="same"))
    model.add(Conv2D(16, 3, activation='relu', bias_initializer='he_normal', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2
    model.add(Conv2D(32, 3, activation='relu', bias_initializer='he_normal', padding="same"))
    model.add(Conv2D(32, 3, activation='relu', bias_initializer='he_normal', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3
    model.add(Conv2D(32, 3, activation='relu', bias_initializer='he_normal', padding="same"))
    model.add(Conv2D(32, 3, activation='relu', bias_initializer='he_normal', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 全結合層
    model.add(Flatten())
    model.add(Dense(288, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(288))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


model = DeepNeuralNetwork()
# print(model.summary())
# exit()
f_model = './model'
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
# cp_cb = ModelCheckpoint(filepath = os.path.join(f_model,'cnn_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
history = model.fit(x=x_train, y=y_train, epochs=20, batch_size=100, shuffle=True, validation_split=0, callbacks=[early_stopping])

score0 = model.evaluate(x_train, y_train, verbose=0)
print('Train score:', score0[0])
print('Train accuracy:', score0[1])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('save the architecture of a model')
json_string = model.to_json()
open(os.path.join(f_model, 'cnn_model.json'), 'w').write(json_string)
yaml_string = model.to_yaml()
open(os.path.join(f_model, 'cnn_model.yaml'), 'w').write(yaml_string)
print('save weights')
model.save_weights(os.path.join(f_model, 'cnn_model_weights.hdf5'))
