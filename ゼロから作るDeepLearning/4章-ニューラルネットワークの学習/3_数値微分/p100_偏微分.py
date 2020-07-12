import numpy as np
from p98_数値微分 import numerical_diff

def function_2(x):
    if x.ndim == 1:
        return np.sum(x * x)
    else:
        return np.sum(x * x, axis=1)

# x[0]=3, x[1]=4 の時,偏微分df/dx[0]を求めよ.
def function_tmp1(x):
    return x * x + 4 * 4

print(numerical_diff(function_tmp1,3))


# x[0]=3, x[1]=4 の時,偏微分df/dx[1]を求めよ.
print(numerical_diff(lambda x: 9 + x * x, 4))


#-----------------------------------------------------
# 描画

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
def func(x, y):
    return x * x + y * y


# 範囲とプロット回数
x = np.arange(-3.0, 3.0, 0.1)
y = np.arange(-3.0, 3.0, 0.1)

# 二次元メッシュを作る
X, Y = np.meshgrid(x, y)
Z = func(X, Y)


# figureで2次元の図を生成
# Axes3Dで3次元にする
fig = plt.figure()
ax = Axes3D(fig)

# 軸ラベル
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")

# 描画
ax.plot_wireframe(X, Y, Z)
plt.show()