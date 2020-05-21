from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU


#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])  
y = array([4, 5, 6, 7]) 



print("x.shape : ", x.shape)    # (4,3)
print("y1.shape : ", y.shape)  # (4, )    



x = x.reshape(x.shape[0], x.shape[1], 1)   #x.shape :  (4, 3, 1)



print(x.shape) 


# 2. 모델 구성

model = Sequential()
model.add(GRU(10, activation='relu', input_shape = (3,1)))  

# model.add(LSTM(5, input_length = 3, input_dim = 1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))

model.add(Dense(1))
'''
model.summary()


Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru_1 (GRU)                  (None, 10)                360
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 421
Trainable params: 421
Non-trainable params: 0
_________________________________________________________________

LSTM 유닛은 4번씩 내부연산, GRU 유닛은 3번씩 내부연산 

'''


# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, callbacks= [ealry_stopping])

x_predict = array([5,6,7])
x_predict = x_predict.reshape(1,3,1,)

print(x_predict) 


y_predict = model.predict(x_predict)
print(y_predict)       # [[8.012418]]
print(y_predict.shape)   # (1,1)




