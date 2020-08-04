
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

print(x_train.shape)  #(60000, 784)
print(x_test.shape)   #(10000, 784)      # Dense 모델에 맞게 reshape 



# ____________모델 구성____________________

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

input_img = Input(shape =(784,))
encoded = Dense(32, activation ='relu')(input_img)
decoded = Dense(784, activation = 'sigmoid')(encoded)

autoencoder = Model(input_img,decoded)

autoencoder.summary()

autoencoder.compile(optimizer ='adam', loss ='binary_crossentropy')
autoencoder.compile(optimizer ='adam', loss ='mse')
    

autoencoder.fit(x_train,x_train, epochs=30, batch_size = 256, validation_split=0.2)

encoded_imgs = autoencoder.predict(x_test)


import matplotlib.pyplot as plt 

n = 10 
plt.figure(figsize=(20,4))

for i in range(n) : 
    ax = plt.subplot(2, n, i +1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i +1 +n)
    plt.imshow(encoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

# X의 값을 압축시켰다가 다시 펼친것 

'''

# 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 15, batch_size= 60, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)

'''