
# 1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11, 12, 13]) 




# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x,y,epochs=20, batch_size= 1)

'''
metrics = ['mse'] = 수치가 계속 작아짐  
metrics = ['acc'] = 수치가 점점 올라가 1.0에 가까워지고 결국엔 1.0이 된다. 


'''



#  4. 평가, 예측
loss,mse = model.evaluate(x,y) 
print("loss : ", loss)
print("mse : ", mse) 

'''

훈련한 데이터로 평가까지 하고 있다  -> 제대로된 평가 불가
훈련 때는 잘나오던 지표가 실제로 테스트하게 되면 정확도가 떨어질 수 있음 -> keras06_train_test.py

'''


y_pred = model.predict(x_pred) 
print("y_pred : ", y_pred)

'''
loss :  0.3184058666229248
mse :  0.3184058666229248
y_pred :  
[11.983281]
 [13.07    ]
 [14.156715]
 
 '''