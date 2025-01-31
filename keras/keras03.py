
#  ctrl + / = 전체 줄 주석처리
#  ctrl +c  후 ctrl v 전체 줄 복사 및 붙여넣기


# 1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])




# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(5000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x,y,epochs=20, batch_size= 2)


#  4. 평가, 예측
loss,acc = model.evaluate(x,y) #모델의 결과값을 loss 와 acc 라는 변수로 받겠다
print("loss : ", loss)
print("acc : ", acc)








