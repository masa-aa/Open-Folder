import matplotlib.pyplot as plt
from matplotlib.image import imread
img = imread("ゼロから作るDeepLearning/dataset/lena.png")  # 画像の読み込み ワークスペースの ゼロから->dataset -> lena.png
print(img)
plt.imshow(img)
plt.show()