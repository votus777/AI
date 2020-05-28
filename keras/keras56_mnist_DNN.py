
# keras56_mnist_DNN.py

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
print(x_test.shape)   #(10000, 784)      # Dense 모델에 맞게 reshape 



# ____________모델 구성____________________

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten

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

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 15, batch_size= 60, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)

'''

acc 0.98 이상 만들어보자 

loss : 0.2551563577165257
accuracy :  0.930400013923645


loss : 0.1354699911292294
accuracy :  0.9648000001907349


loss : 0.08137337629969117
accuracy :  0.9799000024795532

'''

