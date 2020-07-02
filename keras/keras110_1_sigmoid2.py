

# 1. 데이터 
import numpy as np
import tensorflow as tf
x_train = np.array([1,2,3,4,5,6,7,8,9,10])

y_train = np.array([1,0,1,0,1,0,1,0,1,0])

# x 데이터 하나로 한 개의 회귀와 한 개의 분류 output을 뽑아본다 


# 2. 모델 구성

from keras.models import Sequential, Model
from keras.layers import Dense, Input


model = Sequential()
model.add(Dense(100, input_dim = 1 ))      # activation 명시 없어도 default 존재 
model.add(Dense(1, activation='sigmoid'))
          

# 3. 컴파일, 훈련 

model.compile(loss = ['binary_crossentropy'], optimizer='adam', metrics=['acc'])

model.fit(x_train,y_train, epochs=100, batch_size=1, verbose=1 )

# 4. 평가 및 예측

loss = model.evaluate(x_train,y_train)
print('loss : ', loss)

x_pred = np.array([11,12,13,14])
y_pred = model.predict(x_pred)

print('y_pred : ', y_pred)


'''
https://github.com/votus777/AI_study/wiki/12%EC%9D%BC%EC%B0%A8-%5B-Activation-Function-%5D

'''