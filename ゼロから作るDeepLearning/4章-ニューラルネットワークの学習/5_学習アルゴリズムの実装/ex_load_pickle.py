import sys, os
sys.path.append('c:\\Users\\masay\\OneDrive\\ドキュメント\\project_GALLERIA\\ゼロから作るDeepLearning')
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))  # PILのfromarrayメソッドによって、配列の各値を1byte整数型(0〜255)として画像のpixel値に変換する
    pil_img.show()



if __name__ == '__main__':
    load_network = pickle.load(open("ゼロから作るDeepLearning/dataset/network.sav", 'rb'))
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

    img = x_train[0]
    label = t_train[0]
    print(label)

    print(np.argmax(load_network.predict(img)))
    # img = img.reshape(28, 28)
    # img_show(img)