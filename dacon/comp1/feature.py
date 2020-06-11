
import numpy as np
import pandas as pd
import pywt

import matplotlib.pyplot as plt
import seaborn as sns

from keras import regularizers
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



# 노이즈 제거   wavelet transform

bx = np.array(train.iloc[ : , 1 :36])   # src
(ca, cd) = pywt.dwt(bx, "haar")
cat = pywt.threshold(ca, np.std(ca), mode="soft")
cdt = pywt.threshold(cd, np.std(cd), mode="soft")
x = pywt.idwt(cat, cdt, "haar")

bt = np.array(test.iloc[ : , 1 :36])   # test
(ca, cd) = pywt.dwt(bt, "haar")
cat = pywt.threshold(ca, np.std(ca), mode="soft")
cdt = pywt.threshold(cd, np.std(cd), mode="soft")
tx = pywt.idwt(cat, cdt, "haar")


y = np.array(train.iloc[ : , 71: ])


print(x.shape)  # (10000, 36)
print(tx.shape)  # (10000, 36)
print(y.shape)   # (10000, 4)



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
pca = PCA(n_components=15)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
tx = pca.fit_transform(tx)





# model = DecisionTreeRegressor()
model = RandomForestRegressor(n_estimators=400, max_depth=5)
# model = GradientBoostingRegressor(n_estimators= 400, max_depth=5)
# model = XGBRegressor()
# model = MultiOutputRegressor(XGBRegressor()).fit(x_train, y_train)

model.fit(x_train,y_train)


score = model.score(x_test, y_test)


print(score)



y_predict = model.predict(tx)
print(y_predict)

import matplotlib.pyplot as plt
import numpy as np
# def plot_feature_importance(model):
#     n_features = x.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), x.columns)
#     plt.xlabel("Feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)

feature_importance = model.feature_importances_

# plot_feature_importance(model)

df_fi = pd.DataFrame({'columns':train.columns, 'importances':feature_importance})
df_fi = df_fi[df_fi['importances'] > 0] # importance가 0이상인 것만 
df_fi = df_fi.sort_values(by=['importances'], ascending=False)

plt.show()


# # y_predict = y_predict.to_csv('./data/dacon/comp1/predict.csv', columns=['hhb','hbo2','ca','na'])
# predict = pd.DataFrame(y_predict, columns=['hhb','hbo2','ca','na'])
# predict.index = np.arange(10000,20000)
# predict.to_csv('./data/dacon/comp1/predict.csv', index_label='id')

'''
-0.06276878715135026
[[7.865673  4.157942  9.829988  3.7314901]
 [9.429175  3.6689537 9.845722  3.5462458]
 [8.841846  4.1704874 9.250312  3.6230345]
 ...
 [6.7489786 3.896154  8.566179  2.5990634]
 [7.9405255 4.837055  8.217487  3.06604  ]
 [7.8745785 4.166957  9.035891  3.1297312]]

 '''