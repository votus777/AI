
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense, LSTM
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold, cross_validate

# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

# print(train.shape)         # (10000, 75)   : x_train, x_test, y_train, y_test  
# print(test.shape)          # (10000, 71)   : x_predict   71개의 컬럼으로
# print(submission.shape)    # (10000, 4)    : y_predict    아래 나머지 4개의 컬럼 (hhb, hbo2, ca, na) 을 맟춰라


# print(type(train))         # <class 'pandas.core.frame.DataFrame'>


train = train.interpolate()  # 보간법  // 선형보간 
test = test.interpolate() 
# print(train.isnull().sum())  
# print(test.isnull().sum())  



train = train.fillna(method = 'bfill')   
test = test.fillna(method = 'bfill')



x_train = train.iloc[ :8000 , : 71]
x_test = train.iloc [ :2000 ,  : 71 ]

y_train = train.iloc [ : 8000 , 71 : ]
y_test = train.iloc [ : 2000 , 71: ]

print(x_train.shape)  # (8000, 70)
print(x_test.shape)  # (2000, 70)

print(y_train.shape)  # (8000, 4)
print(y_test.shape)   # (2000, 4)


from sklearn.preprocessing import StandardScaler, MinMaxScaler 

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)
test = standard_scaler.transform(test)



# # 차원 축소

from sklearn.decomposition import PCA
from keras.metrics import mae
pca = PCA(n_components=30)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
test = pca.transform(test)

x_train = x_train.reshape(8000, 30, 1)
x_test = x_test.reshape(2000, 30, 1)
test = test.reshape(10000,30,1)

# 모델

model = Sequential()

model.add(LSTM(10,input_shape=(30,1), activation='relu'))
model.add(Dense(12, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(24, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(12, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))



# 훈련
early_stopping = EarlyStopping(monitor='loss', patience= 20, mode ='auto')
kfold = KFold(n_splits=5, shuffle=True) 

model.compile (optimizer='adam', loss = 'mae', metrics=['mae'])
hist = model.fit(x_train,y_train, verbose=1, batch_size=20,  epochs= 1000, callbacks=[early_stopping], use_multiprocessing=True, validation_split=0.25)


# 평가 및 예측

loss, mse = model.evaluate(x_test,y_test, batch_size=1)
print('loss : ', loss)
print('mae : ', mae )




y_predict = model.predict(test)

print("y_predict : ", y_predict)


import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])   
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.plot(hist.history['val_loss'])

plt.title('loss & mae')
plt.xlabel('epoch')
plt.ylabel('loss,mae')
plt.legend(['train loss', 'test loss', 'train mae', 'test mae'])
plt.show()






# summit file 생성
# y_predict = y_predict.to_csv('./data/dacon/comp1/predict.csv', columns=['hhb','hbo2','ca','na'])
predict = pd.DataFrame(y_predict, columns=['hhb','hbo2','ca','na'])
predict.index = np.arange(10000,20000)
predict.to_csv('./data/dacon/comp1/predict.csv', index_label='id')
