
# keras52_mnist1_imshow.py

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

# mnist 안에 처음부터 train 과 test 가 분리되어 있다. 

# print('x_train : ', x_train[0])
# print('y_train : ', y_train[0])

print(x_train.shape)  #(60000, 28, 28)
print(y_train.shape)  #(60000,)    

print(x_test.shape)   #(10000, 28, 28)
print(y_test.shape)   #(10000,)     


print(x_train[0].shape)   # (28, 28)

plt.imshow(x_train[0], 'gray')  #-> imshow가  28 * 28 행렬 데이터를 받아서 출력 


plt.show()  # img 출력 