import numpy as np
def cross_entropy_error(y, t):
    return - np.sum(t * np.log(y + 1e-7)) # log(0)=-inf なのを注意して実装する


if __name__ == '__main__':
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(cross_entropy_error(y, t))

    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(cross_entropy_error(y, t))