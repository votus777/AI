
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

#3. 훈련

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(x,y, epochs=100, batch_size=1) # x,y를 피트니스 센터에 보내서 한개씩 잘라서 100번 반복 훈련시킨다

#4. 평가 예측
loss, acc = model.evaluate(x,y, batch_size=1)
print("acc: ",acc)
