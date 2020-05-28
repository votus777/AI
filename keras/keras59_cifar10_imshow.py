
# keras59_cifar10.py

# 10가지 컬러 이미지 

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

print('x_train[0] : ', x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()




