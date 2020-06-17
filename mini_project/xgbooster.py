import numpy as np
import pandas as pd

from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


from xgboost  import XGBRFRegressor, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

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

x,y = split_xy(gangnam_velocity,1,1)
        


print("=============================")
print(x)
print(y)
print(x.shape)
print(y.shape) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = False  , train_size = 0.8  
)

print(x_train.shape)  # (216, 1, 4)
print(y_train.shape)  # (216, 1, 4) 
print(x_test.shape)  #(55, 1, 4)

x_train = x_train.reshape(216,4)
x_test = x_test.reshape(55,4)

y_train = y_train.reshape(216,4)
y_test = y_test.reshape(55,4)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)



# 모델 구성

# model = XGBRegressor(max_depth=4)
model = MultiOutputRegressor(XGBRegressor(random_state=12, n_estimators = 500, max_depth = 30)).fit(x_train, y_train)

# 훈련 
model.fit(x_train,y_train)


y_pred = model.predict(x_test)

score = model.score(y_test, y_pred)



print("score : ", score)


# 평가 및 예측



# loss, mse = model.evaluate(x_test, y_test, batch_size=1)

x_predict = np.array([[25.55,26.09,25.9,20.76]])
x_predict = scaler.fit_transform(x_predict)

y_real = np.array([30.42, 23.52, 25.74, 28.62])
y_predict = model.predict(x_predict)

# print("loss : ", loss)
# print("mse : ", mse)
print("y_predict : ",y_predict)
print("y_real : ", y_real )

