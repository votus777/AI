
# keras56_mnist_DNN.py

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

print(x_train.shape)  #(60000, 28, 28)

print(y_train.shape)  #(60000,)    

print(x_test.shape)   #(10000, 28, 28)
print(y_test.shape)   #(10000,)     


print(x_train[0].shape)   # (28, 28)

# plt.imshow(x_train[0], 'gray')  
# plt.show()  


# _________데이터 전처리 & 정규화 _________________

# from keras.utils import np_utils

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# print(y_train.shape)  # (60000, 10) -> one hot encoding 

# 비지도학습에서 x - x 이니까 one hot 할 필요 없음

x_train  = x_train.reshape(60000,784).astype('float32')/255.0    
x_test  = x_test.reshape(10000,784).astype('float32')/255.0  

X =np.append(x_train, x_test, axis =0)

print(X.shape) # (70000, 784)


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)

best_n_components = np.argmax(cumsum >= 0.95) + 1 

print(best_n_components) # 154


