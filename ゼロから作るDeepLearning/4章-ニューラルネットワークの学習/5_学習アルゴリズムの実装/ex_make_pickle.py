import sys, os
sys.path.append('c:\\Users\\masay\\OneDrive\\ドキュメント\\project_GALLERIA\\ゼロから作るDeepLearning')
import numpy as np
from dataset.mnist import load_mnist
from p113_2層のニューラルネットワークのクラス import TwoLayerNet
import pickle


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []


    # ハイパーパラメータ
    iters_num = 10000  # 勾配法による更新回数
    train_size = x_train.shape[0]
    batch_size = 100  # バッチサイズ
    learning_rate = 0.1
    
    iter_per_epoch = max(train_size / batch_size, 1)


    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    for _ in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 勾配の計算 次章で高速化
        # grad = network.numerical_gradient(x_batch, t_batch) # 80時間くらいかかる
        grad = network.gradient(x_batch, t_batch)  # 1分くらい

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if _ % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)


    # モデルを保存する
    filename = 'ゼロから作るDeepLearning/dataset/network.sav'
    pickle.dump(network, open(filename, 'wb'))
    print("created",filename.split("/")[-1])