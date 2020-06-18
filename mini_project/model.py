import numpy as np
import pandas as pd

from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense , LSTM, Dropout, BatchNormalization, BatchNormalization, Conv1D, Flatten
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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

# scaler = RobustScaler()
# gangnam_velocity = scaler.fit_transform(gangnam_velocity)


def split_xy (seq, time_steps, y_col) :
    x,y = list(), list()
    for i in range(len(gangnam_velocity)) :
        x_end_numder = i + time_steps
        y_end_numder = x_end_numder + y_col

        if y_end_numder > len(gangnam_velocity):
            break
    
        tmp_x = gangnam_velocity[i:x_end_numder, :]
        tmp_y = gangnam_velocity[x_end_numder:y_end_numder, :]
        x.append(tmp_x)
        y.append(tmp_y)
    
    return np.array(x),np.array(y)

x,y = split_xy(gangnam_velocity,3,1)
        


print("=============================")
print(x)
print(y)
print(x.shape)
print(y.shape) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = False  , train_size = 0.8  
)

print(x_train.shape)  # (215, 3, 4)
print(y_train.shape)  # (215, 1, 4)
 
print(x_test.shape)

x_train = x_train.reshape(215,12)
x_test = x_test.reshape(54,12)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(215,3,4)
x_test = x_test.reshape(54,3,4)

y_train = y_train.reshape(215,4)
y_test = y_test.reshape(54,4)



# 모델 구성

model = Sequential()

model.add(Conv1D(24,2, activation='relu', input_shape = (3,4) ))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(12, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(6, activation='relu'))
model.add(BatchNormalization())



model.add(Dense(4))


# 훈련 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience=5,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse', metrics=['mse'])
model.fit( x_train, y_train, epochs=10000, callbacks= [ealry_stopping], batch_size=1, validation_split=0.2)

# 평가 및 예측



loss, mse = model.evaluate(x_test, y_test, batch_size=1)

x_predict = np.array([[[28.65,24.03,25.7,22.43],
                      [24.28,25.94,23.67,19.95],
                      [25.55,26.09,25.9,20.76]],
                      [[25.09,23.12,19.5,21.56],
                      [28.65,24.03,25.7,22.43],
                      [24.28,25.94,23.67,19.95]]])


y_predict = model.predict(x_predict)
y_real = [[30.42, 23.52, 25.74, 28.62],[25.55,26.09,25.9,20.76]]


print("loss : ", loss)
print("mse : ", mse)
print("y_predict : ",y_predict)
print("y_real : ", y_real )

from sklearn.metrics import r2_score
r2 = r2_score(y_real, y_predict)

print("R2 score : ", r2)


test_predictions = model.predict(x_train).flatten()

plt.scatter(y_train, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-10, 10], [-10, 10])

plt.show()