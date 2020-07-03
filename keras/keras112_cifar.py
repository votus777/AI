

'''
from keras.datasets import cifar100, cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


import matplotlib.pyplot as plt
import tensorflow as tf

# 데이터 

(x_train, y_train),(x_test,y_test) = cifar10.load_data()


# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (50000, 100) -> one hot encoding 

x_train  = x_train.reshape(50000,32,32,3).astype('float32')/255.0  
x_test  = x_test.reshape(10000,32,32,3).astype('float32')/255.0  


sin = tf.math.sin

# ____________모델 구성____________________


model= Sequential()

model.add(Conv2D(64, (3,3), activation = sin, padding = 'same',input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))

model.add(Conv2D(64, (3,3), activation = sin, padding = 'same' ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))

model.add(Conv2D(128, (3,3), activation = sin, padding = 'same' ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))

model.add(Conv2D(32, (3,3), activation = sin, padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))

model.add(Flatten())
model.add(Dense(100, activation= sin))
model.add(Dense(10, activation= 'softmax'))


model.summary()



# 훈련 



early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])

model.compile(optimizer=Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics=['acc'] )
                            # = 0.0004          이걸 쓰게 되면 one hot encoding 안써도 된다

model.fit(x_train,y_train, epochs= 20, batch_size= 240, validation_split= 0.2 , callbacks= [early_stopping])




# 평가 및 예측 


loss, acc = model.evaluate(x_train,y_train)
val_loss, val_acc = model.evaluate(x_test, y_test)
  
print('loss :', loss)
print('accuracy : ', acc)
print('val_loss :', loss)
print('val_accuracy : ', acc)


# 시각화


# plt.figure(figsize= (10,6))



# plt.subplot(2, 1, 1)    

# plt.plot(history['loss'] , marker = '.', c = 'red', label = 'loss')  
# plt.plot(history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')  
# plt.grid()
# plt.title('loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['loss', 'val_loss'])   
# plt.show()



# plt.subplot(2, 1, 2)   

# plt.plot(history['acc'])
# plt.plot(history['val_acc'])
# plt.grid()
# plt.title('accuracy')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.legend(['acc', 'val_acc'])
# plt.show()
'''

# keras70을 카피해서 Sequential로 바꾸고, cifar10으로,
# sparse_categorical_crossentropy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import Adam

sin = tf.math.sin

# 클래스 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
# cp = ModelCheckpoint(filepath = modelfath, monitor = 'val_loss',
#                      mode = 'auto', save_best_only = True)
# tb_hist = TensorBoard(log_dir = './graph', histogram_freq = 0,
#                       write_graph = True, write_images = True)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)            # (50000, 32, 32, 3)
print(x_test.shape)             # (10000, 32, 32, 3)
print(y_train.shape)            # (50000, 1)
print(y_test.shape)             # (10000, 1)

# 1-1. 정규화
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0

# 1-2. OHE
# y_train = np_utils.to_categorical(y_train, num_classes = 100)
# y_test = np_utils.to_categorical(y_test, num_classes = 100)
# print(y_train.shape)
# print(y_test.shape)


# 2. 모델링

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                 input_shape = (32, 32, 3), padding = 'same',
                 activation = sin))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                 padding = 'same', activation = sin))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = sin))
model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                 padding = 'same', activation = sin))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = sin))
model.add(Conv2D(filters = 128, kernel_size = (3, 3),
                 padding = 'same', activation = sin))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Flatten())
model.add(Dense(256, activation = sin))
model.add(Dense(10, activation = 'softmax'))

model.summary()


# 3. 컴파일 및 훈련
model.compile(loss = 'sparse_categorical_crossentropy',         # 원핫인코딩을 하지 않았을 때, 다중분류 손실함수
              metrics = ['accuracy'],                           # sparse는 개인 취향이다!
              optimizer = Adam(1e-4))                           # 0.0001
hist = model.fit(x_train, y_train,
                 epochs = 20, batch_size = 120,
                 validation_split = 0.3, verbose = 1)

print(hist.history.keys())


# 4. 모델 평가
res = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", res[0])
print("acc : ", res[1])


# 5. 시각화
plt.figure(figsize = (10, 6))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.title('loss')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['accuracy'], marker = '.', c = 'violet', label = 'acc')
plt.plot(hist.history['val_accuracy'], marker = '.', c = 'green', label = 'val_acc')
plt.title('accuracy')
plt.grid()
plt.ylim(0, 1.0)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc = 'lower right')
plt.show()


