
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
from sklearn.model_selection import KFold, cross_validate

from xgboost import XGBRegressor, XGBModel

from sklearn.multioutput import MultiOutputRegressor
from pandas.plotting import scatter_matrix

from sklearn.model_selection import KFold, cross_val_score


# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)





# 노이즈 제거   wavelet transform


train = train.interpolate(method='values') 
test = test.interpolate(method='values')

train = train.fillna(method = 'bfill') 
test = test.fillna(method = 'bfill') 



bx = np.array(train.iloc[ : , 1 :71])   # src
(ca, cd) = pywt.dwt(bx, "haar")
cat = pywt.threshold(ca, np.std(ca), mode="soft")
cdt = pywt.threshold(cd, np.std(cd), mode="soft")
x = pywt.idwt(cat, cdt, "haar")

bt = np.array(test.iloc[ : , 1 :71])   # test
(ca, cd) = pywt.dwt(bt, "haar")
cat = pywt.threshold(ca, np.std(ca), mode="soft")
cdt = pywt.threshold(cd, np.std(cd), mode="soft")
tx = pywt.idwt(cat, cdt, "haar")





y = np.array(train.iloc[ : , 71: ])


print(y[-10 : -1])

# print(x.shape)  # (10000, 36)
# print(tx.shape)  # (10000, 36)
# print(y.shape)   # (10000, 4)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x,y, shuffle = True  , train_size = 0.8  
)

# 표준화
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)
tx = standard_scaler.fit_transform(tx)

# 차원 축소

from sklearn.decomposition import PCA
from keras.metrics import mae
pca = PCA(n_components=36)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
tx = pca.fit_transform(tx)



# 모델


model = MultiOutputRegressor(XGBRegressor(learning_rate = 0.1, max_depth = 10,n_estimators = 1000))


# 훈련
early_stopping = EarlyStopping(monitor='loss', patience= 20, mode ='auto')
kfold = KFold(n_splits=10, shuffle=True) 

# model.compile (optimizer='adam', loss = 'mae', metrics=['mae'])
hist = model.fit(x_train,y_train)

#  verbose=1, batch_size=10,  epochs= 1000, callbacks=[early_stopping], use_multiprocessing=True, validation_split=0.25

# 평가 및 예측

# loss, mse = model.evaluate(x_test,y_test, batch_size=1)
score = model.score(x_test,y_test)
# print('loss : ', loss)
# print('mae : ', mae )




y_predict = model.predict(tx)

print("y_predict : ", y_predict)
print(y_test)
predict = pd.DataFrame(y_predict, columns=['hhb','hbo2','ca','na'])
predict.index = np.arange(10000,20000)
predict.to_csv('./data/dacon/comp1/predict.csv', index_label='id')
