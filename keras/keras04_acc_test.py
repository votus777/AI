
# 1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([110000000, 120000000, 130000000]) #x_predict




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
model.fit(x,y,epochs=20, batch_size= 1)


#  4. 평가, 예측
loss,acc = model.evaluate(x,y) #모델의 결과값을 loss 와 acc 라는 변수로 받겠다
print("loss : ", loss)
print("acc : ", acc) #평가

y_pred = model.predict(x_pred) #이제 훈련을 시켰으니 x_pred의 값을 예측해 보겠다
print("y_pred : ", y_pred)

'''
y_pred :  
 [10.993746]
 [11.993542]
 [12.993331]

acc=1.0 이지만 예측값이 11,12,13인 정확한 데이터가 나오질 않는다!

X_pred에 11,12,13이 아니라 11000000,1200000,13000000을 넣어보면 오차값을 더욱 잘 볼 수 있다. 
절대 무시할만한 값이 아니다!

모델에 허점이 있는 것일까? 왜 이런 오차가 발생할까? -> mse는 예측값과 실제값의 차이의 제곱의 평균

또한 loss는 있는데 왜 acc는 1이였을까? ->  metrics=['acc'] 여기서 acc는 분류지표이기 때문이다. 

분류 지표인 acc는 y값이 고정되어 있지만 mse는 고정되어 있지 않기에 이런 문제가 나타난다.  

11.0001이 나와서 acc는 사실은 0이겠지만 실제로는 분류방식인 ["acc"]에서 11로 판단하고 acc=1.0을 줘버린다. 

loss를 따라가야 하는데 갑자기 분류 acc를 따라가서 발생하는 문제  -> keras05_mse.py

'''
