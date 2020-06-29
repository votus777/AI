import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten 

from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor, LGBMClassifier 



epm = pd.read_csv( 'AI_2020\\task19\\train_data\\train_EPM.csv', header = 0, index_col = 0)
swe = pd.read_csv('AI_2020\\task19\\train_data\\train_SWE.csv', header = 0, index_col='time_tag', parse_dates=True)
xray = pd.read_csv('AI_2020\\task19\\train_data\\train_xray.csv', header = 0, index_col='time_tag', parse_dates=True)
proton = pd.read_csv('AI_2020\\task19\\train_data\\train_proton.csv', header = 0, index_col = 0)

xray.info()

# 결측치 '-100'을 nan으로 바꾸기
epm = epm.replace(-100 , float("nan"))
swe = swe.replace(-100 , float("nan"))
xray = xray.replace(-100 , float("nan"))
proton = proton.replace(-100 , float("nan"))

# 결측치 보간 
epm = epm.interpolate(method = 'linear') 
swe = swe.interpolate(method = 'linear') 
xray = xray.interpolate(method = 'linear') 
proton = proton.interpolate(method = 'linear') 

epm = epm.fillna(epm.mean())
swe = swe.fillna(swe.mean())
xray = xray.fillna(xray.mean())
proton = proton.fillna(proton.mean())

print(swe.head())


# 다운 샘플링 
swe = swe.resample('5min').mean()
xray = xray.resample('5min').mean()

# print(epm.shape)      # (782974, 8)
# print(swe.shape)      # (799488, 2)
# print(xray.shape)     # (799488, 2)
# print(proton.shape)   # (799488, 1)

# Numpy 변환 
epm = np.array(epm.iloc [ : , 3:5 ])
swe = np.array(swe.iloc[ : 782974])
xray = np.array(xray.iloc [ : 782974])
proton = np.array(proton.iloc [ : 782974])

# print(epm.shape)      # (782974, 2)
# print(swe.shape)      # (782974, 2)
# print(xray.shape)     # (782974, 2)
# print(proton.shape)   # (782974, 1)



x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
epm, swe, xray, proton, train_size = 0.8, shuffle=True, random_state = 24)


# kfold = KFold(n_splits=5, shuffle=True, random_state = 41)  


# model = XGBRegressor(learning_rate= 0.01, n_estimators=800, 
#                         colsample_bytree = 0.8, n_jobs = -1, objective = 'regression',  
#                         boosting_type='gbdt').fit([x1_train,x2_train,x3_train], y_train)



# 2. 모델 구성____________________________
from keras.models import Sequential, Model  
from keras.layers import Dense, Input 



#model -------- 1
input1 = Input(shape=(2, ), name= 'input_1') 

dense1_1 = Dense(2400, activation= 'relu', name= '1_1') (input1) 
dense1_2 = Dense(1200,activation='relu', name = '1_2')(dense1_1)
dense1_3 = Dense(2400,activation='relu', name = '1_3')(dense1_2)


#model -------- 2
input2 = Input(shape=(2, ), name = 'input_2') 

dense2_1 = Dense(2400, activation= 'relu', name = '2_1')(input1) 
dense2_2 = Dense(1200, activation='relu', name = '2_2')(dense2_1)
dense2_3 = Dense(2400, activation='relu', name = '2_3')(dense2_2)

#model -------- 3
input3 = Input(shape=(2, ), name = 'input_3') 

dense3_1 = Dense(2400, activation= 'relu', name = '3_1')(input1) 
dense3_2 = Dense(1200, activation='relu', name = '3_2')(dense3_1)
dense3_3 = Dense(2400, activation='relu', name = '3_3')(dense3_2) 


#이제 두 개의 모델을 엮어서 명시 
from keras.layers.merge import concatenate    #concatenate : 사슬 같이 잇다
merge1 = concatenate([dense1_3, dense2_3,dense3_3], name = 'merge') #파이썬에서 2개 이상은 무조건 list []

middle1 = Dense(120, activation='relu')(merge1)
middle1 = Dense(120)(middle1)




################# output 모델 구성 ####################

output1 = Dense  (80,activation='relu',name = 'output_1')(middle1)
output1_2 = Dense (120, activation='relu',name = 'output_1_2')(output1)
output1_3 = Dense (80, activation='relu', name = 'output_1_3')(output1_2)
output1_4 = Dense (1, name = 'output_1_4')(output1_3)



model = Model (inputs = [input1, input2, input3], outputs= (output1_4))



# 3. 훈수 두는 훈련_______________________________________________________________________________

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 10, mode ='auto')


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train, x3_train], 
          y_train, epochs=50, batch_size = 1000, validation_split= 0.25, verbose = 1, callbacks=[early_stopping])  #2개 이상은 모두 []로 묶어준다


#4. 평가, 예측____________________________________________
loss = model.evaluate([x1_test,x2_test,x3_test], y_test, batch_size = 1000 ) # 여기도 역시 묶어준다

print(loss)



y_predict = model.predict([x1_test,x2_test,x3_test]) # 리스트의 함정에 조심!!!

for i in range(len(y_predict)) :
    print(y_test[i], y_predict[i])




#________RMSE 구하기_________________________________________
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ", (RMSE(y_test, y_predict)))



#________R2 구하기_____________________
from sklearn.metrics import r2_score

def R2(y_test, y_predict) :
    return r2_score(y_test, y_predict)

print("R2 score : ", (R2(y_test,y_predict)))
# _____________________________________


