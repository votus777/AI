

# 1. 데이터 
import numpy as np
import tensorflow as tf
x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])


y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

# compile parameter를 추가해보자   loss_weights

# 2. 모델 구성

from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape = (1, ))

x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape = (1, ))

x2 = Dense(5)(input2)
x2 = Dense(5)(x2)
x2 = Dense(5)(x2)

merge = concatenate([x1,x2])

# 회귀 모델 
x3 = Dense(100)(merge)
x3 = Dense(100)(x3)
output1 = Dense(1)(x3)


# 분류 모델 
x4 = Dense(5)(merge)
x4 = Dense(5)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs = [input1,input2] , outputs = [output1, output2])

# model.summary()

# 3. 컴파일, 훈련 

model.compile(loss = ['mse', 'binary_crossentropy'], 
              optimizer='adam', 
              metrics=['mse', 'acc'],
              loss_weights=[0.01, 0.99])  # 회귀는 너무 잘 맞는데 분류가 잘 맞지 않으니 분류모델에 좀 더 비중을 두자 

model.fit([x1_train,x2_train],[y1_train,y2_train],epochs=50, batch_size=1, verbose=1 )

# 4. 평가 및 예측

loss = model.evaluate([x1_train,x2_train],[y1_train,y2_train])
print('loss : ', loss)

x1_pred = np.array([11,12,13,14])
x2_pred = np.array([11,12,13,14])

y_pred = model.predict([x1_pred,x2_pred])

print('y_pred : ', y_pred)

'''
loss :  [0.7099474668502808, 0.056574732065200806, 0.7165471911430359, 0.056574732065200806, 1.0, 0.2614303231239319, 0.5] 
                 전에는 2,3번쨰 합치면 1번째 총loss가 나왔는데 여기서는 0.1, 0.9 비중을 다르게 주었기 때문에 단순 합이 아니다 
y_pred :  [array([[10.577348],
       [11.538765],
       [12.500185],
       [13.461605]], dtype=float32), array([[0.60927856],
       [0.61003757],
       [0.6107957 ],
       [0.6115536 ]], dtype=float32)]

그러나 슬프게도 전혀 나아지지 않는다.  그냥 따로 쓰자 
       
'''
