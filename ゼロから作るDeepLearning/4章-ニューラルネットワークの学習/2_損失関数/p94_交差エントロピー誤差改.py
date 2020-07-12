import numpy as np


def cross_entropy_error(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return - np.sum(t * np.log(y + 1e-7)) / batch_size
    
def cross_entropy_error2(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # tがbool値であることに注目する. t==1であるところだけ欲しい.
    return - np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    