import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.1)
    plt.show()
