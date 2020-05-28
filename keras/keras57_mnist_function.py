
# keras57_mnist_function.py

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

print(x_train.shape)  #(60000, 28, 28)

print(y_train.shape)  #(60000,)    

print(x_test.shape)   #(10000, 28, 28)
print(y_test.shape)   #(10000,)     


print(x_train[0].shape)   # (28, 28)

# plt.imshow(x_train[0], 'gray')  
# plt.show()  


# _________데이터 전처리 & 정규화 _________________

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10) -> one hot encoding 

x_train  = x_train.reshape(60000,784).astype('float32')/255.0  
x_test  = x_test.reshape(10000,784).astype('float32')/255.0  

print(x_train.shape)  #(60000, 784)
print(x_test.shape)   #(10000, 784)      # functional 모델에 맞게 reshape 



# ____________모델 구성____________________

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input

model= Sequential()

input1 = Input(shape=(784,), name= 'input_1') 


dense1 = Dense  (512, activation = 'relu', name = 'output_1')(input1)
batch_1 = BatchNormalization()(dense1)

dense1_2 = Dense (512, activation = 'relu',  name = 'output_1_2')(batch_1)
batch_2 = BatchNormalization()(dense1_2)


dense1_3 = Dense (1024, activation = 'relu', name = 'output_1_3')(batch_2)
batch_3 = BatchNormalization()(dense1_3)
dropout = Dropout(0.25)(batch_3)

dense1_4 = Dense (256, activation = 'relu', name = 'output_1_4')(dropout)
batch_4 = BatchNormalization()(dense1_4)


dense1_5 = Dense (10, activation= 'softmax' , name = 'output_1_5')(batch_4)


model = Model (inputs = input1, outputs= (dense1_5))


model.summary()



# 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 20, batch_size= 60, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)

'''

acc 0.98 이상 만들어보자 


loss : 0.22726470713244165
accuracy :  0.9760000109672546


loss : 0.14187905727395164
accuracy :  0.9805999994277954

loss : 0.09247951093358527
accuracy :  0.9824000000953674

'''
