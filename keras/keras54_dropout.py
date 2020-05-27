
# 
# keras54_dropout.py

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

x_train  = x_train.reshape(60000,28,28,1).astype('float32')/255.0   # CNN 모델에 input 하기 위해 4차원으로 만들면서 실수형으로 형변환 & 0과 1 사이로 Minmax정규화 
x_test  = x_test.reshape(10000,28,28,1).astype('float32')/255.0      

# ____________모델 구성____________________

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

model= Sequential()

model.add(Conv2D(32, (2,2), activation = 'relu', input_shape = (28, 28, 1)))   #output ->  (28, 28, 10)
model.add(Conv2D(64, (2,2), activation = 'relu', strides= 1))  
 
model.add(MaxPool2D(pool_size= 2))                               

model.add(Conv2D(32, (2,2),activation = 'relu')) 
model.add(Dropout(0.25))     #  이 위에 있는 노드들의 25%를 drop out 하겠다 

model.add(Flatten())

model.add(Dense(32, activation='relu'))


# model.add(Dense(10, activation= 'softmax')) 
model.add(Dense(10, activation= 'softmax')) 



model.summary()


# 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 20, batch_size= 12, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)
