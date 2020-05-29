<<<<<<< HEAD

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np 



# keras65_fashion_DNN_sqeuntial.py


# 데이터 

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000, 784).astype('float32')/ 255.0
x_test = x_test.reshape(10000, 784).astype('float32')/ 255.0


# 모델 구성


model= Sequential()

model.add(Dense(256, activation = 'relu', input_shape = (784,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))


model.add(Dense(512,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10, activation= 'softmax')) 

model.summary()



# 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 15, batch_size= 60, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)


'''

loss : 0.5810819535128883
accracy :  0.8575999736785889



'''
=======

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np 



# keras65_fashion_DNN_sqeuntial.py


# 데이터 

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000, 784).astype('float32')/ 255.0
x_test = x_test.reshape(10000, 784).astype('float32')/ 255.0


# 모델 구성


model= Sequential()

model.add(Dense(256, activation = 'relu', input_shape = (784,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))


model.add(Dense(512,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10, activation= 'softmax')) 

model.summary()



# 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 15, batch_size= 60, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)


'''

loss : 0.5810819535128883
accracy :  0.8575999736785889



'''
>>>>>>> 76a0c1a03bf932446afe7e9d0ad24529001de4f5
