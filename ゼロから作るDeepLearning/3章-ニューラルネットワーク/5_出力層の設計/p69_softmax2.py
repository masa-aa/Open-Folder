import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)


if __name__ == '__main__':
    a = np.array([1010, 1000, 990])
    print(np.sum(softmax(a)))
