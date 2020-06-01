
# from keras91_load_checkpoint.py 

import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


################ 데이터 불러오기 #####################

x_train = np.load('./data/mnist_train_x.npy')
x_test = np.load('./data/mnist_test_x.npy')


y_train = np.load('./data/mnist_train_y.npy')
y_test = np.load('./data/mnist_test_y.npy')

#####################################################



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


hist = model.fit(x_train,y_train, epochs= 15, batch_size= 100, validation_split= 0.25 ,callbacks= [early_stopping])

'''
########################################################################################################################################
model.save('./model/model_test01.h5')  # fitting 한 다음에 save -> fit 한 결과값( 가중치 )가 저장된다 -> load를 하면 가중치 그대로 출력
#######################################################################################################################################

'''

from keras.models import load_model

model = load_model('./model/14 -  0.0264.hdf5')


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)
# val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  


print('loss :', loss)
print('accuracy : ', acc)
 
