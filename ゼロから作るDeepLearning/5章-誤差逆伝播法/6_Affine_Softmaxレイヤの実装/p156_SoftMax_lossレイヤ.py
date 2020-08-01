import numpy as np
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../..")
from common.functions import cross_entropy_error,softmax

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 損失
        self.y = None  # softmaxの出力
        self.t = None  #教師データ(one-hot vector)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx