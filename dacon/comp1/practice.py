
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

src = train.iloc[ :, 1]
dst = train.iloc[ : , 36]

print(src.shape)
print(dst.shape)

concat = pd.concat([src,dst], axis=1)   # 650_src & 650_Dst 합쳐줌

concat = concat.dropna(axis=0)  #결측치 있는 값 제거 
print(concat.shape)

x = concat.iloc[ : , 0]
y = concat.iloc[ : , 1]

print(x.head())
print(y.head())


kfold = KFold(n_splits=5, shuffle=True)

model = XGBRegressor( n_estimators=300, cv=5, n_jobs=6 )

scores = cross_val_score(model, x, y, cv= kfold)



# predict = concat[concat['650_dst'].isin([''])]
# score = model.score(y_test, y_pred)

# print(predict.shape)