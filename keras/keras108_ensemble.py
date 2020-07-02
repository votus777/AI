

# 1. 데이터 
import numpy as np
import tensorflow as tf
x_train = np.array([1,2,3,4,5,6,7,8,9,10])

y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

# x 데이터 하나로 한 개의 회귀와 한 개의 분류 output을 뽑아본다 


# 2. 모델 구성

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape = (1, ))

x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

#=======분기점===========

# 회귀 모델 
x2 = Dense(100)(x1)
x2 = Dense(100)(x2)
output1 = Dense(1)(x2)


# 분류 모델 
x3 = Dense(10)(x1)
x3 = Dense(10)(x3)
output2 = Dense(1, activation='sigmoid')(x3)

model = Model(inputs = input1 , outputs = [output1, output2])

# model.summary()

# 3. 컴파일, 훈련 

model.compile(loss = ['mse', 'binary_crossentropy'], optimizer='adam', metrics=['mse', 'acc'])

model.fit(x_train,[y1_train,y2_train],epochs=50, batch_size=1, verbose=1 )

# 4. 평가 및 예측

loss = model.evaluate(x_train,[y1_train,y2_train])
print('loss : ', loss)

x_pred = np.array([11,12,13,14])
y_pred = model.predict(x_pred)

print('y_pred : ', y_pred)


'''
10/10 - 0s 2ms/step - loss: 0.7194 - dense_6_loss: 1.7045e-04 - dense_9_loss: 0.7192 - dense_6_mse: 1.7045e-04 - dense_6_acc: 1.0000 - dense_9_mse: 0.2626 - dense_9_acc: 0.4000

loss :  [0.7141631841659546, 0.034767843782901764, 0.679395318031311, 0.034767843782901764, 0.8999999761581421, 0.24324429035186768, 0.6000000238418579]
            총 loss               회귀모델 loss



y_pred :  [array([[ 9.915436],[10.117756],[10.14246 ], [10.046953]], dtype=float32), 

array([[0.35255837],[0.33165494],[0.31828701],[0.31227687]], dtype=float32)]  -> weight 쏠림 현상 

'''

