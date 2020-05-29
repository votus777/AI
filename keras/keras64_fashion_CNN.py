

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np 

# keras64_fashion_CNN_sqeuntial.py




# 데이터 

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28, 1).astype('float32')/ 255.0
x_test = x_test.reshape(10000, 28, 28, 1 ).astype('float32')/ 255.0


# 모델 구성


model= Sequential()

model.add(Conv2D(32, (5,5),activation = 'relu',padding= 'same', input_shape = (28, 28, 1 )))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3),activation = 'relu',input_shape = (28, 28, 1 )))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(MaxPool2D(2,2))

model.add(Conv2D(512, (3,3),activation = 'relu',padding= 'same', input_shape = (28, 28, 1 )))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Conv2D(32, (3,3),activation = 'relu',padding= 'same', input_shape = (28, 28, 1 )))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (2,2),activation = 'relu', input_shape = (28, 28, 1 )))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(MaxPool2D(2,2))


model.add(Conv2D(16, (3,3),activation = 'relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPool2D(2,2))


model.add(Flatten())

model.add(Dense(64,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Dense(10, activation= 'softmax')) 

model.summary()



# 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 15, batch_size= 120, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)

'''

loss : 0.32225194731294104
accuracy :  0.9050999879837036

'''