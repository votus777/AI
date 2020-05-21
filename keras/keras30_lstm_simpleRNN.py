from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])  
y = array([4, 5, 6, 7]) 



print("x.shape : ", x.shape)    # (4,3)
print("y1.shape : ", y.shape)  # (4, )    



x = x.reshape(x.shape[0], x.shape[1], 1)   #x.shape :  (4, 3, 1)



print(x.shape) 


# 2. 모델 구성

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape = (3,1)))  

model.add(SimpleRNN(12, input_length = 3, input_dim = 1))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


'''
model.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn_1 (SimpleRNN)     (None, 10)                120
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 88
_________________________________________________________________
dense_2 (Dense)              (None, 10)                90
_________________________________________________________________
dense_3 (Dense)              (None, 10)                110
_________________________________________________________________
dense_4 (Dense)              (None, 10)                110
_________________________________________________________________
dense_5 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 6
=================================================================
Total params: 579
Trainable params: 579
Non-trainable params: 0
_________________________________________________________________

LSTM 유닛은 4번씩 내부연산, GRU 유닛은 3번씩 내부연산, simpleRNN은 1번씩 




'''




# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 70,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, callbacks= [ealry_stopping])

x_predict = array([5,6,7])
x_predict = x_predict.reshape(1,3,1,)

print(x_predict) 


y_predict = model.predict(x_predict)
print(y_predict)        
print(y_predict.shape)   # (1,1)



