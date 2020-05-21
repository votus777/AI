from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 함수형 모델로 리뉴얼 

#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11], [10, 11, 12], 
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])  

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) 


x = x.reshape(13, 3, 1)
# x = x.reshape(x.shape[0], x.shape[1], 1)


# 2. 모델 구성


input1 = Input(shape=(3,1)) 

dense = (LSTM(10, activation='relu', input_shape = (3,1)))(input1)  


dense1 =(Dense(5))(dense)
dense1 =(Dense(5))(dense1)
dense1 =(Dense(5))(dense1)
dense1 =(Dense(5))(dense1)
dense1 =(Dense(5))(dense1)

output1 = (Dense(1))(dense1)

model = Model(inputs=input1, outputs = output1)

# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, callbacks= [ealry_stopping])


x_input  = array([5,6,7]) 
x_input = x_input.reshape(1,3,1)


yhat = model.predict(x_input)
print(yhat)        # [[7.9967585]] [[8.008283]]
print(yhat.shape)   # (1,1)






