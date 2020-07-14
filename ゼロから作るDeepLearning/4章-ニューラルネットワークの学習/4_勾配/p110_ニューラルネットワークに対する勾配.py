import sys, os
sys.path.append('c:\\Users\\masay\\OneDrive\\ドキュメント\\project_GALLERIA\\ゼロから作るDeepLearning')
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  #ガウス分布で初期化
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

if __name__ == '__main__':
    net = simpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)

    print(np.argmax(p))

    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    dW = numerical_gradient(lambda w: net.loss(x, t), net.W) # ここの挙動注意 loss(x,t)はwに関して定数ではなくnet.Wの関数なのでちゃんと動く
    print(dW)