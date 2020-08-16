import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../..")
import numpy as np
from common.util import im2col, col2im

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + (H + 2*self.pad - FH) // self.stride
        out_w = 1 + (W + 2*self.pad - FW) // self.stride

        col = im2col(x, FH, FW, self.stride, self.pad) # 短冊を作成
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b # dotをすることで(短冊にしたので)アダマール積の和が取れる
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  # N,OH,OW,FN -> N,FN,OH,OW 

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx