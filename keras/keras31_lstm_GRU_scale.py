from numpy import array
from keras.models import Sequential
from keras.layers import Dense, GRU


#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11], [10, 11, 12], 
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])  

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) 



x_predict = array([50,60,70])



x = x.reshape(x.shape[0], x.shape[1], 1)   #x.shape :  (13, 3, 1)
x_predict = x_predict.reshape(1,3,1)



# 2. 모델 구성

model = Sequential()
model.add(GRU(10, activation='relu', input_shape = (3,1)))  

# model.add(LSTM(10, input_length = 3 ,input_shape = 1))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, callbacks= [ealry_stopping], verbose= 2)


y_predict = model.predict(x_predict)
print(y_predict)      #[[80.06215]] [[80.45786]] 





