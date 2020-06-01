
# keras53_mnist2.py

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
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization

model= Sequential()

model.add(Conv2D(64, (3,3), input_shape = (28, 28, 1), padding='same'))   
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size= 2))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3,3), input_shape = (28, 28, 1), padding='same'))   
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size= 2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size= 2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())                               


model.add(Dense(10, activation= 'softmax')) 



model.summary()

# 딥러닝 텐서플로 케라스 교재 201p 


model.save('./model/sample/mnist/model_mnist.h5') 

# 훈련 


from keras.callbacks import EarlyStopping, ModelCheckpoint 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

modelpath = './model/sample/mnist{epoch:02d} - {val_loss: .4f}.hdf5' 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)

model.fit(x_train,y_train, epochs= 15, batch_size= 120, validation_split= 0.25 ,callbacks= [early_stopping,checkpoint])


model.save_weights('./model/sample/mnist/mnist_weight1.h5')

# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)
