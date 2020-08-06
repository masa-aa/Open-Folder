import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../..")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 検証データの分離
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=1.865405500969014e-05)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='AdaGrad', optimizer_param={'lr': 0.002737364082615975})
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list

test_acc_list, train_acc_list = __train(15)



# グラフの描画========================================================


plt.ylim(0.1, 1.03)
x = np.arange(len(test_acc_list))
plt.plot(x, test_acc_list)
plt.plot(x, train_acc_list, "--")


plt.show()