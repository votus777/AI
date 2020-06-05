
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

import seaborn as sns

'''
# 데이터 불러오기 

wine_datasets = pd.read_csv("./data/winequality-white.csv", index_col = 0, header = 1, sep = ';', encoding = 'ISO-8859-1')
                                   
print(wine_datasets.shape)      # (4897, 11)

# for i in range(len(wine_datasets.index)) :
#     for j in range(len(wine_datasets.iloc[i])):
#         wine_datasets.iloc[i,j] = int (wine_datasets.iloc[i,j].replace(',',''))


# wine_datasets = wine_datasets.sort_values(['일자'], ascending = [True])


wine_datasets = wine_datasets.values

print(type(wine_datasets))   # <class 'numpy.ndarray'>



# wine_datasets = wine_datasets.astype(int).astype(float)

np.save('./data/winequality.csv',arr=wine_datasets)


'''
# 데이터 구성

wine = np.load('./data/winequality.npy', allow_pickle=True)



# print(wine[0])
# print(wine.shape)   # (4897, 11)


wine_x = wine[ :-21 , :10]
wine_y = wine [ :-21 , -1: ]

wine_x_predict = wine[-20 : , : 10]
wine_y_predict = wine[-20 : , -1 : ]

# print(wine_x.shape)  # (4876, 10)
# print(wine_y.shape)  # (4876, 1)   (3 ~ 9)


wine_y= np_utils.to_categorical(wine_y)               

# print(wine_y.shape)  # (4876, 10)
# print(wine_y[0])   #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]

wine_y = wine_y [ : , 3:]

# print(wine_y[0])   #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]



from sklearn.model_selection import train_test_split
wine_x_train, wine_x_test, wine_y_train, wine_y_test = train_test_split(
   
    wine_x, wine_y, shuffle = True  , train_size = 0.8  
)


standard_scaler = StandardScaler()
wine_x_train = standard_scaler.fit_transform(wine_x_train)
wine_x_test = standard_scaler.fit_transform(wine_x_test)


pca = PCA(n_components=5)
wine_x_train = pca.fit_transform(wine_x_train)
wine_x_test = pca.fit_transform(wine_x_test)


# print(wine_x_train.shape)  # (3900, 5)
# print(wine_x_test.shape)   # (976, 5)

# print(wine_y_train.shape)   # (3900, 7)
# print(wine_y_test.shape)    # (976, 7)


# print(wine_x_predict.shape)  # (20, 10)
# print(wine_y_predict.shape)  # (20, 1)


# 모델 구성


model = RandomForestClassifier(n_estimators= 500)



# 훈련 

model.fit(wine_x_train, wine_y_train)


# 예측 및 평가 

y_predict = model.predict(wine_x_test)

acc = accuracy_score(wine_y_test,y_predict)  
score = model.score(wine_x_test,wine_y_test) # model.score = model.evaluate


print("acc : ", acc)
print("score : ", score)


'''
acc :  0.5307377049180327
score :  0.5307377049180327


- 왜 acc가 낮게 나올까..?

y 값이 너무 한 군데로 몰려있다. ( 5, 6 이 넘쳐난다)

만약 5,6이 80% 이상을 차지한다면, 그냥 5,6만 찍어도 acc 80 나오는것 -> 데이터 전처리 필요 





'''


