import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,6,0.1) # 0<=k<6で0.1刻みで生成
y = np.sin(x)
plt.plot(x,y)
plt.show()