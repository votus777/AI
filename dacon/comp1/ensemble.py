
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from keras.metrics import mae
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense, LSTM, LeakyReLU, Input, Concatenate
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

train = train.interpolate(method='values') 
test = test.interpolate(method='values') 

src_columns = [k for k in train.columns if 'src' in k]
dst_columns = [k for k in train.columns if 'dst' in k]

src_columns_t = [k for k in test.columns if 'src' in k]
dst_columns_t = [k for k in test.columns if 'src' in k]


train_src = train[src_columns]
train_dst = train[dst_columns]

test_src = test[src_columns_t]
test_dst = test[dst_columns_t]


train[src_columns] = train_src.interpolate(method = 'linear',axis=1)
train[dst_columns] = train_dst.interpolate(method = 'linear',axis=1)

test[dst_columns] = test_dst.interpolate(method = 'linear' ,axis=1)
test[src_columns] = test_src.interpolate(method = 'linear', axis=1)


train[dst_columns] = train_dst.fillna(method = 'bfill')   
test[dst_columns] = test_dst.fillna(method = 'bfill')

train[src_columns] = train_src.fillna(method = 'bfill')   
test[src_columns] = test_src.fillna(method = 'bfill')


train[dst_columns] = train_dst.fillna(train_dst.mean())   
test[dst_columns] = test_dst.fillna(test_dst.mean())

train[src_columns] = train_src.fillna(train_src.mean())   
test[src_columns] = test_src.fillna(test_src.mean())





train_x_dst = train.loc[:, '650_dst':'990_dst']
train_x_src = train.loc[:, '650_src' : '990_src']
y = train.loc[:, 'hhb':'na']
print(train_x_dst.shape)  # (10000, 35)
print(y.shape)  # (10000, 4)


from sklearn.model_selection import train_test_split
x_train_dst, x_test_dst,x_train_src, x_test_src, y_train, y_test = train_test_split(
   
    train_x_dst,train_x_src, y, shuffle = True  , train_size = 0.8  
)


from sklearn.preprocessing import StandardScaler, MinMaxScaler 

standard_scaler = StandardScaler()
x_train_dst = standard_scaler.fit_transform(x_train_dst)
x_test_dst = standard_scaler.transform(x_test_dst)

x_train_src = standard_scaler.fit_transform(x_train_src)
x_test_src = standard_scaler.transform(x_test_src)

test_dst = standard_scaler.transform(x_test_dst)
test_src = standard_scaler.transform(test_src)

# 차원 축소

from sklearn.decomposition import PCA
from keras.metrics import mae
pca = PCA(n_components=30)
x_train_dst = pca.fit_transform(x_train_dst)
x_train_src = pca.fit_transform(x_train_src)

x_test_dst = pca.transform(x_test_dst)
x_test_src = pca.transform(x_test_src)

test_dst = pca.transform(test_dst)
test_src = pca.transform(test_src)

# 모델


#model -------- 1
input1 = Input(shape=(30, ), name= 'input_1') 

x1 = Dense(32, activation= 'relu', name= '1_1') (input1) 
x1 = Dropout(0.4)(x1)
x1 = Dense(16,activation='relu', name = '1_2')(x1)
x1 = Dropout(0.2)(x1)



#model -------- 2
input2 = Input(shape=(30, ), name = 'input_2') 

x2 = Dense(32, activation= 'relu', name = '2_1')(input2) 
x2 = Dropout(0.4)(x2)
x2 = Dense(16,activation='relu', name = '2_2')(x2)
x2 = Dropout(0.2)(x2)


 


from keras.layers.merge import concatenate    
merge1 = concatenate([x1, x2], name = 'merge') 

middle1 = Dense(24, activation='relu')(merge1)
middle1 = Dropout(0.3)(middle1)
middle1 = Dense(24, activation='relu')(middle1)
middle1 = Dropout(0.3)(middle1)



################# output 모델 구성 ####################

x3 = Dense  (12,activation='relu',name = 'output_1')(middle1)
x3 = Dropout(0.2)(x3)
x3 = Dense (12,activation='relu', name = 'output_1_3')(x3)
x3 = Dense (4,activation='relu', name = 'output_1_4')(x3)



model = Model (inputs = [input1, input2], outputs= (x3))


# 훈련
early_stopping = EarlyStopping(monitor='loss', patience= 10, mode ='auto')
kfold = KFold(n_splits=5, shuffle=True) 
reduce = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=3)

model.compile (optimizer='adam', loss = 'mae', metrics=['mae'])
hist = model.fit([x_train_dst,x_train_src],y_train, verbose=1, batch_size=10, epochs= 200, callbacks=[early_stopping], use_multiprocessing=True, validation_split=0.25)



# 평가 및 예측

loss, mse = model.evaluate([x_test_dst,x_test_src],y_test, batch_size=1)
print('loss : ',loss)
print('mae : ', mae )




y_predict = model.predict([test_dst,test_src])

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

