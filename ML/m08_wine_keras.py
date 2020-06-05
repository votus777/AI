
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import csv

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


# 데이터 구성

wine = np.load('./data/winequality.npy', allow_pickle=True)



print(wine[0])
print(wine.shape)   # (4897, 11)


wine_x = wine[ :-21 , :10]
wine_y = wine [ :-21 , -1: ]

wine_x_predict = wine[-20 : , : 10]
wine_y_predict = wine[-20 : , -1 : ]

print(wine_x.shape)  # (4876, 10)
print(wine_y.shape)  # (4876, 1)   (3 ~ 9)


wine_y= np_utils.to_categorical(wine_y)               

print(wine_y.shape)  # (4876, 10)
print(wine_y[0])   #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]

wine_y = wine_y [ : , 3:]

print(wine_y[0])   #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]



from sklearn.model_selection import train_test_split
wine_x_train, wine_x_test, wine_y_train, wine_y_test = train_test_split(
   
    wine_x, wine_y, shuffle = True  , train_size = 0.8  
)


standard_scaler = StandardScaler()
wine_x_train = standard_scaler.fit_transform(wine_x_train)
wine_x_test = standard_scaler.fit_transform(wine_x_test)


pca = PCA(n_components=4)
wine_x_train = pca.fit_transform(wine_x_train)
wine_x_test = pca.fit_transform(wine_x_test)


print(wine_x_train.shape)  # (3900, 5)
print(wine_x_test.shape)   # (976, 5)

print(wine_y_train.shape)   # (3900, 7)
print(wine_y_test.shape)    # (976, 7)


print(wine_x_predict.shape)  # (20, 10)
print(wine_y_predict.shape)  # (20, 1)


# 모델 구성


model = Sequential()
model.add(Dense(10, activation='relu', input_dim = 4))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='softmax'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))


# 훈련 

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping( monitor='loss', patience= 20, mode ='auto')

model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['acc'])

hist = model.fit(wine_x_train, wine_y_train, epochs= 10000, batch_size= 5, validation_split= 0.25,  callbacks= [early_stopping] )


# 예측 및 평가 

loss, acc = model.evaluate(wine_x_test,wine_y_test, batch_size=1)


print('loss :', loss)
print('accuracy : ', acc)


'''

loss : 1.1615333949200441
accuracy :  0.506147563457489

'''

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])   # plot 추가 =  선 추가 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['val_loss'])

plt.title('loss & acc')
plt.xlabel('epoch')
plt.ylabel('loss,acc')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()
