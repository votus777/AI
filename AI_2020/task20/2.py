import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import time


from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten , BatchNormalization

from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor, LGBMClassifier 

from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.model_selection import KFold, cross_val_score


train = pd.read_csv('./AI_2020/task20/train.csv', header = 1, index_col = [0,1])
val = pd.read_csv('./AI_2020/task20/val.csv', header = 1, index_col = [0,1])
test = pd.read_csv('./AI_2020/task20/test.csv', header = 1, index_col = [0,1])

         

x = train.iloc[ :,  [0, 1, 5, 6, 7, 19, 24, 25, 29, 31]]
y = train.iloc[ : , [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]]

x_val = val.iloc[ :,  [0, 1, 5, 6, 7, 19, 24, 25, 29, 31]]
y_val = val.iloc[ : , [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]]

x_pred = test.iloc[ :,  [0, 1, 5, 6, 7, 19, 24, 25, 29, 31]]
y_pred = test.iloc[ : , [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]]

print(x.shape)  # (2895, 10)
print(y.shape)  # (2895, 25)

x = np.array(x)
y = np.array(y)

x_val = np.array(x_val)
y_val = np.array(y_val)

x_pred = np.array(x_pred)
y_pred = np.array(y_pred)




x_train, x_test, y_train, y_test = train_test_split(
x,y, train_size = 0.8, shuffle=False, random_state = 44)


# from sklearn.decomposition import PCA
# from keras.metrics import mae
# pca = PCA(n_components=2)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
# x_pred = pca.transform(x_pred)


print(x_train.shape)  # (2316, 10)
print(y_train.shape)  # (2316, 25)


start = time.time()


model = XGBRegressor(learning_rate= 0.01, 
                      n_estimators=900, n_jobs = -1, 
                    gpu_id=0, tree_method='gpu_hist', metrics = 'rmsle', objective='reg:squarederror')

model.fit(x_train,y_train)




# from keras.callbacks import EarlyStopping, ModelCheckpoint
# early_stopping = EarlyStopping( monitor='loss', patience= 10, mode ='auto')

# model.compile(loss = 'mse', optimizer='adam', metrics = ['mse'])

# hist = model.fit(x_train,y_train, epochs= 1000, batch_size= 20, validation_data=[x_val,y_val] ,callbacks= [early_stopping])

# loss, mse = model.evaluate(x_test,y_test, batch_size=100)


end = time.time() - start

valiation = model.predict(x_val)


score = model.score(valiation,y_val)
print('점수 : ', score)
print('========================')
print(model.feature_importances_)



y_pred = model.predict(x_pred)

predict = pd.DataFrame(y_pred)
predict = predict.iloc[360: ]
predict.to_csv('./AI_2020/task20/predict_lgbm2.csv', header=0, index=0)


print("총 걸린 시간 : ", end) # 단위 : 초 

