
# 사실은 keras53_mnist2.py

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


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 20, batch_size= 120, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)


'''
loss : 0.1388316419814875
accuracy :  0.9736999869346619

좋게 나온것 같지만 케글을 보면 평균 이하의 accuracy...


(conv relu conv relu pool) * 3


#과제 0.9925 이상 뽑아내기


loss : 0.028588482533274805
accuracy :  0.9908999800682068

loss : 0.029769239512317983
accuracy :  0.9927999973297119


loss : 0.025018695568875583
accuracy :  0.9930999875068665

loss : 0.02020222538380057
accuracy :  0.9939000010490417

여기서 cpu의 한계가 나온다. 겁나 느리다  

gpu로 바꾸니 한결 낫다 그래도 부족하다.. 

'''