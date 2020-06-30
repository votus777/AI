import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

from statsmodels.tsa.arima_model import ARIMA


from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten 

from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor, LGBMClassifier 

from sklearn.multioutput import MultiOutputRegressor


train = pd.read_csv('./AI_2020/task20/train.csv', header = 1, index_col = [0,1])
val = pd.read_csv('./AI_2020/task20/val.csv', header = 1, index_col = [0,1])
test = pd.read_csv('./AI_2020/task20/test.csv', header = 1, index_col = [0,1])

# print("val.shape : ",val.shape)   # (720, 36)
# print("test.shape : ",test.shape) # (720, 36)


# train.iloc[882] = np.array([ 308024, 87728, 8129, 21055, 8147, 5623, 115624, 5936, 13350, 8277, 
#                          60908, 15114, 24214, 14920, 31827, 46209, 8595, 23577, 61015,
#                          89910, 58269, 11297, 11059, 7054, 151508, 7251, 13648, 42689, 21848, 
#                          4529, 19072, 4442, 19843, 11828, 19147])  # 2월 6일 18시 결측치 보간 

# row = train.iloc[2271:2295]

# train = pd.DataFrame(np.insert(train.values, 2125, row, axis=0 )) # 3월 30일을 4월 6일 0 ~ 23값으로 치환 

train = train.drop([2150], axis=0)  
                 

x = train.iloc[ :,  [ 1, 5, 6, 7, 19, 24, 25, 29, 31]]
y = train.iloc[ : , [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]]

x_pred = test.iloc[ :,  [ 1, 5, 6, 7, 19, 24, 25, 29, 31]]
y_pred = test.iloc[ : , [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]]


print(x.shape)  # (2918, 10)
print(y.shape)  # (2918, 25)

x = np.array(x)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(
x,y, train_size = 0.8, shuffle=True, random_state = 39)


print(x_train.shape)  # (2334, 10)
print(y_train.shape)  # (2334, 25)

model = MultiOutputRegressor(LGBMRegressor(learning_rate= 0.01, n_estimators=1500, 
                        colsample_bytree = 0.8, n_jobs = -1,  
                        max_bin =100, boosting_type='gbdt' ) ).fit(x_train, y_train)


y_pred = model.predict(x_pred)

predict = pd.DataFrame(y_pred)
predict = predict.iloc[360: ]
predict.to_csv('.\AI_2020\task20\predict_lgbm.csv', header=0, index=0)
