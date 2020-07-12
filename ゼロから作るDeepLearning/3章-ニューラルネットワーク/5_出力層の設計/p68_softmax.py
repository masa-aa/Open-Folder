import numpy as np


def softmax(a):
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a)


if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    print(softmax(a))

# memo np.array([1010,1000,990])とかだとオーバーフローする
