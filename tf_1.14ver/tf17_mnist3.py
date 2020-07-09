import tensorflow as tf
import numpy as np

from keras.datasets import mnist

# 데이터 

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape) #(60000, 28, 28)
print(y_train.shape) #(60000,)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2]).astype('float32')/255
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2]).astype('float32')/255


x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])

