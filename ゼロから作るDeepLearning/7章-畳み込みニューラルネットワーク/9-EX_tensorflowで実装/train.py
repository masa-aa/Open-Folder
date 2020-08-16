# # load data

from keras.datasets import mnist
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from IPython.core.display import display

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# build model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape



# compare models

# +

class AlexNet_like:
    def __init__(self, hid_dim_0=32, hid_dim_1=64):
        self.input = Input(shape=(28, 28), name='Input')  # shape of input
        self.reshape = Reshape((28, 28, 1), name='Reshape')  # 28x28 -> 28x28x1
        self.layers = {}
        self.layers['conv_0'] = Conv2D(hid_dim_0, 3, name='Conv_0')  # 28x28x1 -> 26x26xhid_dim_0
        self.layers['pool_0'] = MaxPooling2D((2, 2), strides=(1, 1), name='Pool_0')  # 26x26xhid_dim_0 -> 13x13xhid_dim_0
        self.layers['conv_1'] = Conv2D(hid_dim_1, 3, name='Conv_1')  # 13x13xhid_dim_0 -> 11x11xhid_dim_1
        self.layers['pool_1'] = MaxPooling2D((3, 3), strides=(2, 2), name='Pool_1')  # 11x11xhid_dim_1 -> 5x5xhid_dim_1
        self.layers['flatten'] = Flatten()
        self.layers['dense_0'] = Dense(units=128, activation='relu')  # dense + ReLU
        self.last = Dense(units=10, activation='softmax', name='last')
    
    
    def build(self):
        x = self.input
        z = self.reshape(x)
        for layer in self.layers.values():
            z = layer(z)
        p = self.last(z)
        
        model = Model(inputs=x, outputs=p)
        
        return model
# -

dim_hidden_layres = [1<<i for i in range(4, 8)]

# +
df_accuracy = pd.DataFrame()

for hid_dim_0 in dim_hidden_layres:
    for hid_dim_1 in dim_hidden_layres:
        print('========', 'hid_dim_0:', hid_dim_0, '; hid_dim_1:', hid_dim_1, '========')
        model = CNNModel(hid_dim_0=hid_dim_0, hid_dim_1=hid_dim_1)
        model = model.build()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        callbacks = [
            EarlyStopping(patience=3),
            ModelCheckpoint(filepath=os.path.join('model_{}_{}.h5'.format(hid_dim_0, hid_dim_1)), save_best_only=True),
        ]
        n_param = model.count_params()
        model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, callbacks=callbacks, validation_split=0.2)
        acc = accuracy_score(y_test, model.predict(x_test).argmax(axis=-1))
        
        df_accuracy = pd.concat([df_accuracy, pd.DataFrame([[hid_dim_0, hid_dim_1, n_param, acc]], columns=['hid_dim_0', 'hid_dim_1', 'n_param', 'accuracy'])])
# -

display(df_accuracy.set_index(['hid_dim_0', 'hid_dim_1'])[['n_param']].unstack())
display(df_accuracy.set_index(['hid_dim_0', 'hid_dim_1'])[['accuracy']].unstack())

df_accuracy.to_csv('cnn_results.csv')