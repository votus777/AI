
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt

(x_train, y_train), (x_test,y_test) = mnist.load_data()



y_train = y_train [ :1000]
y_test = y_test [ : 1000]



print(x_train.shape)  #(60000, 28, 28)

print(y_train.shape)  #(60000,)    

print(x_test.shape)   #(1000, 28, 28)
print(y_test.shape)   #(1000,)     


print(x_train[0].shape)   # (28, 28)

# plt.imshow(x_train[0], 'gray')  
# plt.show()  



# _________데이터 전처리 & 정규화 _________________

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10) -> one hot encoding 

x_train  = x_train[ : 1000].reshape(-1,28,28,1).astype('float32')/255.0   # CNN 모델에 input 하기 위해 4차원으로 만들면서 실수형으로 형변환 & 0과 1 사이로 Minmax정규화 
x_test  = x_test[ : 1000].reshape(-1,28,28,1).astype('float32')/255.0      

# ____________모델 구성____________________

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization

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

model.add(Conv2D(32, (2,2), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())

model.add(Dropout(0.3))
model.add(Conv2D(32, (2,2), activation = 'relu', strides= 1, padding='same'))  
model.add(BatchNormalization())


model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

 
model.add(BatchNormalization())
model.add(Dense(10, activation= 'softmax')) 



model.summary()


# 훈련 

early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

modelpath = './model/{epoch:02d}-{val_loss: .4f}.hdf5' # 02d : 두자리 정수,  .4f : 소수점 아래 4자리 까지 float 실수

checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, mode = 'auto')

tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph= True, write_images= True)

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])  # metrics = ["accuracy"] 도 가능한데 중요한건 뭐라고 쓰든 뒤에서도 통일해서 써야한다. 
hist = model.fit(x_train,y_train, epochs= 10, batch_size= 120, validation_split= 0.25 ,callbacks= [early_stopping, checkpoint, tb_hist ])


# 평가 및 예측 


loss, acc = model.evaluate(x_train,y_train, batch_size=1)
val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  
print('loss :', loss)
print('accuracy : ', acc)


# 히스토리 

loss = hist.history ['loss']
val_loss = hist.history['val_loss']

acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc : ', acc)
print('val_acc : ', val_acc)

'''

plt.figure(figsize= (10,6))


plt.subplot(2, 1, 1)    # 2행 1열의 첫번쨰 그림을 사용하겠다. 인덱스는 0부터 시작하는데, 이건 아니다.

plt.plot(hist.history['loss'] , marker = '.', c = 'red', label = 'loss')  # plot 추가 =  선 추가 
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')  
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.legend(['loss', 'val_loss'])   # 1st legend : 1st plot,  2nd legend : 2nd plot 
plt.legend(loc='upper right')
plt.show()



plt.subplot(2, 1, 2)    # 2행 1열의 첫번쨰 그림을 사용하겠다. 


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['acc', 'val_acc'])
plt.show()
'''