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
    # load network
    load_network = pickle.load(open("ゼロから作るDeepLearning/dataset/network.sav", 'rb'))

    # import image
    img = np.array(Image.open('ゼロから作るDeepLearning/dataset/seven.png').convert('L')).flatten() / 255.0
    
    # predict
    print(np.argmax(load_network.predict(img)))