import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,6,0.1) # 0<=k<6で0.1刻みで生成
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle="--",label="cos") # 破線で描画
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend() # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=18)
plt.show()