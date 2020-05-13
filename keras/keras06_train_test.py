
# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])


x_pred = np.array([16, 17, 18]) 




# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=30, batch_size = 1)




#  4. 평가, 예측
loss,mse = model.evaluate(x_test, y_test, batch_size = 1) 
print("loss : ", loss)
print("mse : ", mse) 



y_pred = model.predict(x_pred) 
print("y_pred : ", y_pred)

'''

loss 값이 최소로 나올 때까지 하이퍼 파라미터 튜닝을 해보자

Dense (500) 보다 Dense(64)가 더 잘나온다 속도도 빠르고 
layer가 깊다고 항상 좋은건 아닌듯

1.2345e-05 = 0.0000012345 


'''