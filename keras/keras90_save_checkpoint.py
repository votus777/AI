
# from keras85_save_model.py 

import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

(x_train, y_train), (x_test,y_test) = mnist.load_data()

print(x_train.shape)  #(60000, 28, 28)

print(y_train.shape)  #(60000,)    

print(x_test.shape)   #(10000, 28, 28)
print(y_test.shape)   #(10000,)     


print(x_train[0].shape)   # (28, 28)

# plt.imshow(x_train[0], 'gray')  
# plt.show()  



# _________데이터 전처리 & 정규화 _________________



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10) -> one hot encoding 

x_train  = x_train.reshape(60000,28,28,1).astype('float32')/255.0   # CNN 모델에 input 하기 위해 4차원으로 만들면서 실수형으로 형변환 & 0과 1 사이로 Minmax정규화 
x_test  = x_test.reshape(10000,28,28,1).astype('float32')/255.0      

# ____________모델 구성____________________



model= Sequential()

model.add(Conv2D(32, (2,2), input_shape = (28, 28, 1), padding='same'))   #output ->  (28, 28, 10)
model.add(BatchNormalization())

model.add(Conv2D(32, (2,2), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size= 2))
model.add(Dropout(0.2)) 
                             
                               

model.add(Conv2D(64, (2,2), activation = 'relu', padding='same')) 
model.add(BatchNormalization())

model.add(Conv2D(64, (2,2), activation = 'relu', padding='same')) 
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size= 2))
model.add(Dropout(0.3))

model.add(Conv2D(256, (2,2), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())

model.add(Dropout(0.3))
model.add(Conv2D(128, (2,2), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())


model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))


 
model.add(BatchNormalization())
model.add(Dense(10, activation= 'softmax')) 



model.summary()



########################################
# model.save('./model/model_test01.h5')
########################################

# 컴파일, 훈련 




early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])  # metrics = ["accuracy"] 도 가능한데 중요한건 뭐라고 쓰든 뒤에서도 통일해서 써야한다. 

modelpath = './model/{epoch:02d} - {val_loss: .4f}.hdf5' 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)


hist = model.fit(x_train,y_train, epochs= 15, batch_size= 100, validation_split= 0.25 ,callbacks= [early_stopping, checkpoint])


########################################################################################################################################
model.save('./model/model_test01.h5')  # fitting 한 다음에 save -> fit 한 결과값( 가중치 )가 저장된다 -> load를 하면 가중치 그대로 출력
#######################################################################################################################################



# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)
# val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  
print('loss :', loss)
print('accuracy : ', acc)
 

'''

Epoch 00013: val_loss improved from 0.02870 to 0.02688, saving model to ./model/13 -  0.0269.hdf5
Epoch 14/15
45000/45000 [==============================] - 11s 238us/step - loss: 0.0135 - acc: 0.9957 - val_loss: 0.0264 - val_acc: 0.9932

Epoch 00014: val_loss improved from 0.02688 to 0.02642, saving model to ./model/14 -  0.0264.hdf5
Epoch 15/15
45000/45000 [==============================] - 11s 238us/step - loss: 0.0145 - acc: 0.9954 - val_loss: 0.0275 - val_acc: 0.9935

Epoch 00015: val_loss did not improve from 0.02642
10000/10000 [==============================] - 28s 3ms/step
loss : 0.02030830052649761
accuracy :  0.9939000010490417


'''