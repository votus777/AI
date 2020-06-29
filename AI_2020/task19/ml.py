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

# 제로 패딩 



# Numpy 변환 


x = np.concatenate((epm,swe,xray), axis = 1) 
y = proton



kfold = KFold(n_splits=10, shuffle=True, random_state = 41)  


model = XGBRegressor(learning_rate= 0.01, n_estimators=800, 
                colsample_bytree = 0.8, n_jobs = -1, objective = 'reg:squaredlogerror', gpu_id=0, tree_method='gpu_hist') 
                     

scores = cross_val_score(model,x,y, cv=kfold, verbose= 2)

print("====================")
print(scores)
