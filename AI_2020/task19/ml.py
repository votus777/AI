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


epm_train = pd.read_csv( 'AI_2020\\task19\\train_data\\train_EPM.csv', header = 0, index_col = 0)
swe_train = pd.read_csv('AI_2020\\task19\\train_data\\train_SWE.csv', index_col='time_tag', parse_dates=True)
xray_train = pd.read_csv('AI_2020\\task19\\train_data\\train_xray.csv',index_col='time_tag', parse_dates=True)
proton_train = pd.read_csv('AI_2020\\task19\\train_data\\train_proton.csv', header = 0, index_col = 0)

epm_val = pd.read_csv('AI_2020\\task19\\val_data\\val_EPM.csv', header = 0, index_col = 0)
swe_val = pd.read_csv('AI_2020\\task19\\val_data\\val_SWE.csv', index_col='time_tag', parse_dates=True)
xray_val = pd.read_csv('AI_2020\\task19\\val_data\\val_xray.csv', index_col='time_tag', parse_dates=True)
proton_val = pd.read_csv('AI_2020\\task19\\val_data\\val_proton.csv', header = 0, index_col = 0)

epm_test = pd.read_csv('AI_2020\\task19\\test_data\\test_EPM.csv', header = 0, index_col = 0)
swe_test = pd.read_csv('AI_2020\\task19\\test_data\\test_SWE.csv', index_col='time_tag', parse_dates=True)
xray_test = pd.read_csv('AI_2020\\task19\\test_data\\test_xray.csv', index_col='time_tag', parse_dates=True)
proton_test = pd.read_csv('AI_2020\\task19\\test_data\\test_proton.csv', header = 0, index_col = 0)



# 결측치 '-100'을 nan으로 바꾸기
epm_train = epm_train.replace(-100 , float("nan"))
swe_train = swe_train.replace(-100 , float("nan"))
xray_train = xray_train.replace(-100 , float("nan"))
proton_train = proton_train.replace(-100 , float("nan"))

epm_val = epm_val.replace(-100 , float("nan"))
swe_val = swe_val.replace(-100 , float("nan"))
xray_val = xray_val.replace(-100 , float("nan"))
proton_val = proton_val.replace(-100 , float("nan"))

epm_test = epm_test.replace(-100 , float("nan"))
swe_test = swe_test.replace(-100 , float("nan"))
xray_test = xray_test.replace(-100 , float("nan"))
proton_test = proton_test.replace(-100 , float("nan"))


# 결측치 보간 
epm_train = epm_train.interpolate(method = 'linear') 
swe_train = swe_train.interpolate(method = 'linear') 
xray_train = xray_train.interpolate(method = 'linear') 
proton_train = proton_train.interpolate(method = 'linear') 


epm_val = epm_val.interpolate(method = 'linear') 
swe_val = swe_val.interpolate(method = 'linear') 
xray_val = xray_val.interpolate(method = 'linear') 
proton_val = proton_val.interpolate(method = 'linear') 


epm_test = epm_test.interpolate(method = 'linear') 
swe_test = swe_test.interpolate(method = 'linear') 
xray_test = xray_test.interpolate(method = 'linear') 
proton_test = proton_test.interpolate(method = 'linear') 

epm_train = epm_train.fillna(0)
swe_train = swe_train.fillna(0)
xray_train = xray_train.fillna(0)
proton_train = proton_train.fillna(0)

epm_val = epm_val.fillna(0)
swe_val = swe_val.fillna(0)
xray_val = xray_val.fillna(0)
proton_val = proton_val.fillna(0)

epm_test = epm_test.fillna(0)
swe_test = swe_test.fillna(0)
xray_test = xray_test.fillna(0)
proton_test = proton_test.fillna(0)

# 다운 샘플링 
swe_train = swe_train.resample('5min').mean()
xray_train = xray_train.resample('5min').mean()

swe_val = swe_val.resample('5min').mean()
xray_val = xray_val.resample('5min').mean()


swe_test = swe_test.resample('5min').mean()
xray_test = xray_test.resample('5min').mean()


# Numpy array 변형
epm_train = epm_train.values
swe_train = swe_train.values
xray_train = xray_train.values
proton_train = proton_train.values

epm_val = epm_val.values
swe_val = swe_val.values
xray_val = xray_val.values
proton_val = proton_val.values

epm_test = epm_test.values
swe_test = swe_test.values
xray_test = xray_test.values
proton_test = proton_test.values



# 제로 패딩 
epm_train = np.pad(epm_train, ((16514, 0),(0,0)), 'constant', constant_values = 0)
epm_val = np.pad(epm_val, ((22594, 0),(0,0)), 'constant', constant_values = 0)
epm_test = np.pad(epm_test, ((11172, 0),(0,0)), 'constant', constant_values = 0)

epm_val = epm_val[ : -288, : ]
swe_val = swe_val[ : -288, : ]
xray_val = xray_val[ : -288, : ]


epm_test = epm_test[ : -576, : ]
swe_test = swe_test[ : -576, : ]
xray_test = xray_test[ : -576, : ]

# feature 병합 

x_train = np.concatenate((epm_train, swe_train, xray_train), axis = 1) 
x_val   = np.concatenate((epm_val, swe_val, xray_val), axis = 1) 
x_test  = np.concatenate((epm_test ,swe_test, xray_test), axis = 1)

y_train = proton_train
y_val = proton_val
y_test = proton_test


kfold = KFold(n_splits=5, shuffle=True, random_state = 41)  


model = XGBRegressor(learning_rate= 0.01, n_estimators=1000, booster= 'gblinear', 
                colsample_bytree = 0.8, n_jobs = -1, objective = 'reg:squaredlogerror', gpu_id=0, tree_method='gpu_hist').fit(x_train,y_train) 
                     

scores = cross_val_score(model,x_val,y_val, cv = kfold, verbose= 2)

model.save_model("./AI_2020/task19/model")

y_predict = model.predict(x_test)

print("====================")
print(scores)
print(y_predict)

y_predict = pd.DataFrame(y_predict)
y_test = pd.DataFrame(y_test)

y_test = np.append(y_test, y_predict, axis=0)
y_test = pd.DataFrame(y_test[575136:])

y_test.to_csv('./AI_2020/task19/predict.csv', header=0, index=0)

