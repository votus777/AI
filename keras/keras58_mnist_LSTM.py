
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

x_train  = x_train.reshape(60000,1,784).astype('float32')/255.0    # (784,1) // (28,28) // (392,2) // (196,4)   - 모두 가능 
x_test  = x_test.reshape(10000,1,784).astype('float32')/255.0  

print(x_train.shape)  
print(x_test.shape)       # LSTM 모델에 맞게 reshape 



# ____________모델 구성____________________

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input, LSTM

model= Sequential()

input1 = Input(shape=(1,784), name= 'input_1') 


dense1 = LSTM(256, activation = 'relu', name = 'output_1')(input1)
batch_1 = BatchNormalization()(dense1)
dropout1 = Dropout(0.2)(batch_1)


dense1_2 = Dense (32, activation = 'relu',  name = 'output_1_2')(dropout1)
batch_2 = BatchNormalization()(dense1_2)
dropout2 = Dropout(0.2)(batch_2)



dense1_3 = Dense (32, activation = 'relu', name = 'output_1_3')(dropout2)
batch_3 = BatchNormalization()(dense1_3)
dropout3 = Dropout(0.2)(batch_3)

'''
dense1_4 = Dense (32, activation = 'relu', name = 'output_1_4')(dropout)
batch_4 = BatchNormalization()(dense1_4)
'''

dense1_5 = Dense (10, activation= 'softmax' , name = 'output_1_5')(dropout3)


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
loss : 0.1485041831221041     -> (28,28)
accuracy :  0.9526000022888184

loss : 0.07734253021336447
accuracy :  0.9811000227928162  -> (1,784)
   

서로 연관있는 데이터가 아니기 때문에 굳이 LSTM에 뭉텅이로 넣지 않고 1개씩, 784번 넣어주는 것이 효율 좋은 것 같다. 



(784,1)은 시간 겁나 많이 걸린다.  LSTM이니까


loss : 0.13785222204135755   -> (4,196)
accuracy :  0.9617999792098999

'''
