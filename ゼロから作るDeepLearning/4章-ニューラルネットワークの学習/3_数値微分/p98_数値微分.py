import numpy as np
import matplotlib.pylab as plt


# 悪い実装例
def numerical_diffxxx(f, x):
    h = 1e-50
    return (f(x + h) - f(x)) / h
    # 1e-50=0.0になる

# 良い実装例
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
    # 中心差分で計算

def function_1(x):
    return 0.01 * x * x + 0.1 * x


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    tf = tangent_line(function_1, 5)
    y2 = tf(x)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.show()

    print(numerical_diff(function_1, 10))
    