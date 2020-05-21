from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM


#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])  
y = array([4, 5, 6, 7]) 



print("x.shape : ", x.shape)    # (4,3)
print("y1.shape : ", y.shape)  # (4, )    



x = x.reshape(x.shape[0], x.shape[1], 1)   #x.shape :  (4, 3, 1)
'''
                  행         열      몇 개씩 자르는지         반복X100
x의 shape  = (batch_size, timesteps, feature)  # 3D 텐서
               
input_ shape = (timesteps, feature)
input_length = timesteps, input_dim = feature

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

x_predict = array([5,6,7])
x_predict = x_predict.reshape(1,3,1,)

print(x_predict) 


y_predict = model.predict(x_predict)
print(y_predict)        
print(y_predict.shape)   # (1,1)









