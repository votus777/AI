
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.utils import np_utils

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import metrics


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

print(x_train.shape)  # (8000, 71)
print(x_test.shape)  # (2000, 71)

print(y_train.shape)  # (8000, 4)
print(y_test.shape)   # (2000, 4)


from sklearn.preprocessing import StandardScaler 

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)
test = standard_scaler.transform(test)

# # 차원 축소

from sklearn.decomposition import PCA
from keras.metrics import mae
pca = PCA(n_components=20)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
test = pca.transform(test)






# 모델 


def bulid_model(drop=0.5, optimizer = 'adam') :

    inputs = Input(shape=(20, ), name= 'inputs')  # (786,)
    x = Dense(512, activation='relu', name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(4, activation='relu', name= 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['mae'], loss = 'mae')
    return model

def create_hyperparameters() : 
    batches = [100, 150]
    optimizers = [ 'rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)    # start ~ end 사이의 값을 개수만큼 생성하여 배열로 반환합니다.
    return{"batch_size" :  batches, "optimizer": optimizers, "drop" : dropout }  # girdsearch 가 dictionary 형태로 값을 받기 때문에 return도 dict형태로 맞춰준다 


from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  
# keras.wrappers.scikit_learn.py의 래퍼를 통해 Sequential 케라스 모델을 (단일 인풋에 한정하여) Scikit-Learn 작업의 일부로 사용할 수 있습니다.

model = KerasRegressor(build_fn=bulid_model, verbose = 1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train,y_train)

print(search.best_params_)
