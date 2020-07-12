import numpy as np
import matplotlib.pylab as plt
from p47_ステップ関数のグラフ import step_function
from p48_シグモイド関数 import sigmoid

if __name__ == '__main__':
    x = np.arange(-5.0,5.0,0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    plt.plot(x,y1,label="step")
    plt.plot(x,y2,linestyle="--",label="sigmoid") # 破線で描画
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("step & sigmoid")
    plt.legend() # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
    plt.ylim(-0.1,1.1)
    plt.show()