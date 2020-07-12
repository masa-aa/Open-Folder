# 60000枚の画像から無作為に10枚を選択
import sys, os
sys.path.append('c:\\Users\\masay\\OneDrive\\ドキュメント\\project_GALLERIA\\ゼロから作るDeepLearning')
import numpy as np
from dataset.mnist import load_mnist

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    print(x_train.shape)
    print(t_train.shape)

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)  # 0以上, train_size未満からbatch_size個の数字を選び出す.
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    