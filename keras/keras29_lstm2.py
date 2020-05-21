from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM


#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])  
y = array([4, 5, 6, 7]) 



print("x.shape : ", x.shape)    # (4,3)
print("y1.shape : ", y.shape)  # (4, )    



x = x.reshape(x.shape[0], x.shape[0], 1)   #x.shape :  (4, 3, 1)
'''
                  행         열      몇 개씩 자르는지         
x의 shape  = (batch_size, timesteps, feature)
                덩어리 

'''
print("x.shape : ", x.shape)  

'''
print(x.shape) 


# 2. 모델 구성

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape = (3,1)))  

model.add(LSTM(5, input_length = 3, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1))


# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, callbacks= [ealry_stopping])




x_input  = array([5,6,7])  

x_input = x_input.reshape(1,3,1) 


yhat = model.predict(x_input)
print(yhat)        
print(yhat.shape)   # (1,1)



'''






