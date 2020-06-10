
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

from keras.metrics import mae

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense, LSTM, LeakyReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor, XGBModel

from sklearn.multioutput import MultiOutputRegressor
from pandas.plotting import scatter_matrix

from sklearn.model_selection import KFold, cross_val_score
# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)


train = train.interpolate(method='values') 
test = test.interpolate(method='values') 

train = train.fillna(method = 'bfill') 

rho = train.sort_values(by =['rho'], axis=0)


x = rho[["680_src","750_src","850_src","900_src"]]
test =  rho[["680_src","750_src","850_src","900_src"]]

x = x.iloc[ : , :71]
y = rho.iloc [ :, 71: ]

test = test.iloc[ : , :71]

x = x.values
test = test.values

x1 = x[ :2457, :] * 0.8
x2 = x[ 2457 : 4966, :] * 0.9
x3 = x[ 4966 : 7444, : ] * 1.05
x4 = x[ 7444 : 10001, : ] * 1.2

x = np.concatenate((x1,x2,x3,x4), axis=0)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x,y, shuffle = True  , train_size = 0.8  
)



from sklearn.preprocessing import StandardScaler, MinMaxScaler 

stand = StandardScaler()
x_train = stand.fit_transform(x_train)
x_test = stand.transform(x_test)
test = stand.transform(test)


# 모델

model = Sequential()

model.add(Dense(10,input_dim =4, activation='relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dropout(0.4))
# model.add(Dense(40, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(20, activation= 'relu'))
# model.add(Dropout(0.25))
model.add(Dense(4, activation='relu'))



# 훈련
early_stopping = EarlyStopping(monitor='loss', patience= 50, mode ='auto')
kfold = KFold(n_splits=10, shuffle=True) 

model.compile (optimizer='adam', loss = 'mae', metrics=['mae'])
hist = model.fit(x_train,y_train, verbose=1, batch_size=10,  epochs= 1000, callbacks=[early_stopping], use_multiprocessing=True, validation_split=0.25)


# 평가 및 예측

loss, mse = model.evaluate(x_test,y_test, batch_size=1)
print('loss : ', loss)
print('mae : ', mae )







y_predict = model.predict(test)
print(y_predict)

'''
# summit file 생성
# y_predict = y_predict.to_csv('./data/dacon/comp1/predict.csv', columns=['hhb','hbo2','ca','na'])
predict = pd.DataFrame(y_predict, columns=['hhb','hbo2','ca','na'])
predict.index = np.arange(10000,20000)
predict.to_csv('./data/dacon/comp1/predict.csv', index_label='id')

'''