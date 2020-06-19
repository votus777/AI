
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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_validate

from pandas.plotting import scatter_matrix

from xgboost import XGBRegressor, XGBModel
from lightgbm import LGBMRegressor

# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

# pd.concat()

src = train.iloc[ : , 1]
dst = train.iloc[ : , 36]

print(src.shape)
print(dst.shape)

concat_base = pd.concat([src,dst], axis=1)   # 650_src & 650_Dst 합쳐줌

concat = concat_base.dropna(axis=0)  #결측치 있는 값 제거 

print(concat.shape)


x = concat.iloc[ : , 0]
y = concat.iloc[ : , 1]

x = x.replace(0,-1)
y = y.replace(0,-0.001)

x = x.values
y = y.values

x = x.reshape(8052,1)



from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = 'True', random_state = 18)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


kfold = KFold(n_splits=5, shuffle=True)

n_estimators = 1000
learning_rate = 0.001

colsample_bytree = 0.9
colsample_bylevel = 0.9 

max_depth = 5
n_jobs = -1 

model = LGBMRegressor(max_depth=max_depth, learning_rate= learning_rate, 
                            n_estimators=n_estimators, n_jobs = n_jobs, 
                            colsample_bylevel= colsample_bylevel, num_leaves=90, 
                            )




model.fit(x_train,y_train)

scores = cross_val_score(model, x, y, cv= kfold)

# predict = concat[concat['650_dst'].isin([''])]
score = model.score(x_test, y_test)

print(score)

concat.to_csv('./data/dacon/comp1/concat.csv', index_label='id')

# plt.scatter(x_test, y_test)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.ylim()[1]])
# _ = plt.plot([-10, 10], [-10, 10])

# plt.show()