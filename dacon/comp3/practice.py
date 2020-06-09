
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from keras.metrics import mae

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense, Conv1D, Flatten, MaxPool2D, LSTM
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 데이터 

train = pd.read_csv('./data/dacon/comp3/train_features.csv', header = 0, index_col = 0)
target = pd.read_csv('./data/dacon/comp3/train_target.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', header = 0, index_col = 0)

train = train.interpolate()  # 보간법  // 선형보간 
test = test.interpolate() 

train = train.fillna(method = 'ffill')   
test = test.fillna(method = 'ffill')

train_x = train.values
test = test.values

train_x = train_x [1 : , 1 : ]

print(target.shape)  # (2800, 4)

train_x = train_x.reshape(2800, 375, 4)
test = test.reshape(875,375,4)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    train_x, target, shuffle = False  , train_size = 0.8 
)

# print(x_train.shape)  # (2240, 375, 4)
x_train = x_train.reshape(2240, 375, 4)
x_test = x_test.reshape(560, 375, 4)
test = test.reshape(875, 375, 4)

# 모델 

model = Sequential()


model.add(Conv1D(20,2, input_shape= (375,4), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(4, activation='relu'))


# 훈련 

early_stopping = EarlyStopping(monitor='loss', patience= 10, mode ='auto')

model.compile (optimizer='adam', loss = 'mae', metrics=['mae'])
hist = model.fit(x_train,y_train, verbose=1, batch_size=10,  epochs= 1000, callbacks=[early_stopping], use_multiprocessing=True, validation_split=0.25)


# 평가 및 예측

loss, mse = model.evaluate(x_test,y_test, batch_size=1)
print('loss : ', loss)
print('mae : ', mae )




y_predict = model.predict(test)

print("y_predict : ", y_predict)


