import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int) # bool -> int

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x,y)
    plt.ylim(-0.1,1.1) # y軸の範囲を指定
    plt.show()