
# from keras91_load_checkpoint.py 

import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

(x_train, y_train), (x_test,y_test) = mnist.load_data()

print(x_train.shape)  #(60000, 28, 28)

print(y_train.shape)  #(60000,)    

print(x_test.shape)   #(10000, 28, 28)
print(y_test.shape)   #(10000,)     


print(x_train[0].shape)   # (28, 28)

############# 데이터 저장 ##########################
np.save('./data/mnist_train_x.npy', arr =x_train)
np.save('./data/mnist_test_x.npy', arr =x_test)

np.save('./data/mnist_train_y.npy', arr =y_train)
np.save('./data/mnist_test_y.npy', arr =y_test)

###################################################


'''
 
 numpy가 csv 보다 빠름
 단 numpy는 한 번에 한가지 자료형만 받을 수 있다.
 그래서 좀 더 유연한 pandas를 이용할 수 있다.



'''


# plt.imshow(x_train[0], 'gray')  
# plt.show()  



# _________데이터 전처리 & 정규화 _________________



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10) -> one hot encoding 

x_train  = x_train.reshape(60000,28,28,1).astype('float32')/255.0   # CNN 모델에 input 하기 위해 4차원으로 만들면서 실수형으로 형변환 & 0과 1 사이로 Minmax정규화 
x_test  = x_test.reshape(10000,28,28,1).astype('float32')/255.0      



'''

데이터 저장은 전처리 이후, 이전 상관은 없지만 이후 하는 일이 달라진다 




'''