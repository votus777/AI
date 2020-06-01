
# from keras67_mnist_hist.py 

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
model.save('./model/model_test01_before.h5')
########################################

# 컴파일, 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])  # metrics = ["accuracy"] 도 가능한데 중요한건 뭐라고 쓰든 뒤에서도 통일해서 써야한다. 

hist = model.fit(x_train,y_train, epochs= 15, batch_size= 100, validation_split= 0.25 ,callbacks= [early_stopping])


########################################################################################################################################
model.save('./model/model_test01_after.h5')  # fitting 한 다음에 save -> fit 한 결과값( 가중치 )가 저장된다 -> load를 하면 가중치 그대로 출력
#######################################################################################################################################



# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)
# val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  
print('loss :', loss)
print('accuracy : ', acc)
 


'''
loss : 0.02036984902289645
accuracy :  0.9940000176429749

'''

'''
# 히스토리 

loss_acc = model.evaluate(x_test, y_test)

loss = hist.history ['loss']
val_loss = hist.history['val_loss']

acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc : ', acc)
print('val_acc : ', val_acc)
print('loss_acc :', loss_acc)



import matplotlib.pyplot as plt

plt.figure(figsize= (10,6))



plt.subplot(2, 1, 1)    # 2행 1열의 첫번쨰 그림을 사용하겠다. 인덱스는 0부터 시작하는데, 이건 아니다.

plt.plot(hist.history['loss'] , marker = '.', c = 'red', label = 'loss')  # plot 추가 =  선 추가 
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')  
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])   # 1st legend : 1st plot,  2nd legend : 2nd plot 
# plt.legend(loc='upper right')
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