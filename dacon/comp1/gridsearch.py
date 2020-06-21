
import numpy as np
import pandas as pd
import pywt
import math
import matplotlib.pyplot as plt
import seaborn as sns


from keras import regularizers
from keras.metrics import mae
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense, LSTM, LeakyReLU, Input, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error

from pandas.plotting import scatter_matrix

from xgboost import XGBRegressor as xg
from lightgbm import LGBMRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score

from sklearn.metrics import mean_absolute_error


from lightgbm import LGBMRegressor, LGBMClassifier

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import warnings ; warnings.filterwarnings('ignore')
import time
from sklearn.metrics import f1_score, roc_auc_score, classification_report

# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)


# 노이즈 제거   wavelet transform


train = train.interpolate(method='values') 
test = test.interpolate(method='values')

train = train.fillna(method = 'bfill') 
test = test.fillna(method = 'bfill') 



x = np.array(train.iloc[ : , 36 : 71])
y = np.array(train.iloc[ : , 74 ])


# print(y[-10 : -1])

# print(x.shape)  # (10000, 36)
# print(tx.shape)  # (10000, 36)
# print(y.shape)   # (10000, 4)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x,y, shuffle = True  , train_size = 0.8
)

# 표준화

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# 모델


base_params =  {"n_estimators" : [500,1000 ], "learning_rate" : [ 0.001, 0.01, 0.1],
                 "max_depth" : [4,5,6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9],"gamma" : [0.1, 0.5, 0.9], "n_jobs" : [-1],
                 "objective":['reg:squarederror',  'reg:squaredlogerror'],"random_state":[1],
                 "subsample" : [ 0.7, 0.8, 0.9], "reg_alpha" : [5,6,7]} 
                         

xgb = xg()             
    
grid = GridSearchCV(xgb, base_params, cv =5, n_jobs= -1 ) 
    
        
grid.fit(x_train, y_train)   
    
y_pred = grid.predict(x_test)
    
score  =  r2_score(y_test,y_pred)
    
    
print(grid.best_estimator_)
print('========================')
print(grid.best_params_)

print('========================')
           


# Feature Importance 추출 
# thresholds = np.sort(model.feature_importances_)

# print(thresholds)
'''
# Select from Model 
for thresh in thresholds :     
    
    
    parameters = [ {"n_estimators" : [ 100, 200,400], "learning_rate" : [ 0.0001,0.001, 0.01, 0.1], "max_depth" : [4,5,], "n_jobs" : [-1],
                     "objective":['reg:squarederror',  'reg:squaredlogerror'],"random_state":[1]},
               
               {"n_estimators" : [ 50, 200,400], "learning_rate" : [ 0.001, 0.01, 0.1], 
                "max_depth" : [4,5,6], "colsample_bytree":[0.6, 0.9, 1], "gamma" : [0.1, 0.5, 0.9], "n_jobs" : [-1],
                "objective" :['reg:squarederror',  'reg:squaredlogerror'],"random_state":[1]}, 
               
               {"n_estimators" : [ 90, 200,400 ], "learning_rate" : [ 0.001, 0.01, 0.1],
                 "max_depth" : [4,5,6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9],"gamma" : [0.1, 0.5, 0.9], "n_jobs" : [-1],
                 "objective":['reg:squarederror',  'reg:squaredlogerror'],"random_state":[1]} 
               
              
              ]
  
  
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)                                                                  
    select_x_train = selection.transform(x_train)    
    
    
    xgb = xg()             
    
    grid = GridSearchCV(xgb, parameters, cv =5, n_jobs= -1 ) 
    
        
    grid.fit(select_x_train, y_train)   
    
    select_x_test  = selection.transform(x_test)
    y_pred = grid.predict(select_x_test)
    
    score  =  r2_score(y_test,y_pred)
    
    
    print(grid.best_estimator_)
    print('========================')
    print(grid.best_params_)

    print('========================')
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

print('========================')
# plot_importance(model)    feature, plot 은 xgb 전용
plt.show()

'''