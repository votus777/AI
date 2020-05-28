
# keras59_iris.imshow.py

# 10가지 컬러 이미지 

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np 



(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

print('x_train[0] : ', x_train[0])    # 32 X 32 공간 안에 0 ~ 255 까지의 숫자 배열
print('y_train[0] : ', y_train[0])

print(x_train.shape)  # (50000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)
print(x_test.shape)   # (10000, 32, 32, 3)
print(y_test.shape)   # (10000, 1)

plt.imshow(x_train[0])
plt.show()





