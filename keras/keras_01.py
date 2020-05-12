
#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from keras.models import Sequential #순차적모델
from keras.layers import Dense #기초적인 일차함수

model = Sequential() #요놈을 model이라 부르겠다
model.add(Dense(3, input_dim =1)) #input node는 1개, 1st 히든 레이어 3
model.add(Dense(4)) # 2nd 히든 레이어
model.add(Dense(2)) #3rd "
model.add(Dense(1)) #아웃풋 = credit (예측값)

'''
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 5)                 10
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3
=================================================================
Total params: 39
Trainable params: 39
Non-trainable params: 0

#param = 노드에 연결된 선의 개수 = node*(node_input+1) 
#total param = 총 연산수

#여기서 +1이란, one weight of connection with bias(절편)

'''


#3. 훈련

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(x,y, epochs=100, batch_size=1) # x,y를 피트니스 센터에 보내서 한개씩 잘라서 100번 반복 훈련시킨다

#if batch size > data set의 크기?  - > batchsize default 값 32

#4. 평가 예측
loss, acc = model.evaluate(x,y, batch_size=1)
print("acc: ",acc)


# 그런데 Github 푸시가 안된다 
# 항상 나를 괴롭힌다