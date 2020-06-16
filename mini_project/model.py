import numpy as np
import pandas as pd

from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 데이터 

gangnam_traffic = pd.read_csv('./mini_project/data/강남대로 교통량.csv', header = 0, index_col = 0)
gangnam_velocity = pd.read_csv('./mini_project/data/강남대로 평균속도.csv', header = 0, index_col = 0)
hun_traffic = pd.read_csv('./mini_project/data/헌릉로 교통량.csv', header = 0, index_col = 0)
hun_velocity = pd.read_csv('./mini_project/data/헌릉로 평균속도.csv', header = 0, index_col = 0)
highway_traffic = pd.read_csv('./mini_project/data/용인서울고속도로 교통량.csv', header = 0, index_col = 0)
highway_velocity = pd.read_csv('./mini_project/data/용인서울고속도로 평균속도.csv', header = 0, index_col = 0)


print(gangnam_traffic.shape)  # (272, 4)
print(gangnam_velocity.shape) # (272, 4)
print(hun_traffic.shape)      # (272, 4)
print(hun_velocity.shape)     # (272, 4)
print(highway_traffic.shape)  # (272, 4)
print(highway_velocity.shape) # (272, 4)

gangnam_velocity = gangnam_velocity.values


print(type(gangnam_velocity))

gang_vel_x = gangnam_velocity[ :271 , :3]
gang_vel_y = gangnam_velocity[ 1: , 3]

print(gang_vel_x.shape)
print(gang_vel_y.shape)

gang_vel_x = gang_vel_x.reshape(271, 3, 1)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    gang_vel_x, gang_vel_y, shuffle = False  , train_size = 0.8  
)

print(x_train.shape) 
print(x_test.shape) 
print(y_test.shape)
print(type(x_test))

print(x_test[0])
print(y_test[0])




# 모델 구성

model = Sequential()

model.add(LSTM(20, input_shape= (3,1), activation = 'relu', return_sequences ='False' ))
model.add(LSTM(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))

model.summary()

# 훈련 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 20,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x_train,y_train, epochs=10000, callbacks= [ealry_stopping], batch_size=5)

# 평가 및 예측



# loss, mse = model.evaluate(x_test, y_test, batch_size=5)

x_predict = np.array([28.48, 26.32, 25.75])

x_predict = x_predict.reshape(1,3,1)

y_predict = model.predict(x_predict)



# print("loss : ", loss)
# print("mse : ", mse)

print("y_predict : ", y_predict)

y_predict :  [[23.73917]]
