
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from keras.metrics import mae

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense, LSTM, LeakyReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold, cross_validate

from pandas.plotting import scatter_matrix


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn import preprocessing
from sklearn import utils
from sklearn.multioutput import MultiOutputRegressor


# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)


train = train.interpolate(method='linear')  
test = test.interpolate(method='linear') 


train = train.fillna(method = 'bfill')   
test = test.fillna(method = 'bfill')

# test.filter(regex='_dst$',axis=1).tail().T.plot() 
# test.filter(regex='_src$',axis=1).tail().T.plot() 

# plt.show()

test = test.fillna(0)


x = train.iloc[ : , 1: 71]
y = train.iloc [ :, 71]

# x = x.values
# y = y.values

x_train, x_test ,y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = 0.8, random_state = 12
)




# model = DecisionTreeRegressor()
# model = RandomForestRegressor(n_estimators=400, max_depth=5)
model = GradientBoostingRegressor(n_estimators= 400, max_depth=5)
# model = XGBRegressor(cross_validate=5)
# multioutputregressor = MultiOutputRegressor(XGBRegressor(objective='reg:linear')).fit(x_train, y_train)

model.fit(x_train,y_train)


score = model.score(x_test, y_test)


print(score)


import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importance_cancer(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


plot_feature_importance_cancer(model)
plt.show()
