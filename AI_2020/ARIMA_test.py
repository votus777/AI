import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

from statsmodels.tsa.arima_model import ARIMA


from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten 

from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor, LGBMClassifier 

from sklearn.multioutput import MultiOutputRegressor


from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa import seasonal

train = pd.read_csv('./AI_2020/train.csv', header = 1, index_col = [0,1])
val = pd.read_csv('./AI_2020/val.csv', header = 1, index_col = [0,1])
test = pd.read_csv('./AI_2020/test.csv', header = 1, index_col = [0,1])

test = test.replace(-999, 0)

# print("val.shape : ",val.shape)   # (720, 36)
# print("test.shape : ",test.shape) # (720, 36)


x = train.iloc[ :,  [0, 1, 5, 6, 7, 19, 24, 25, 29, 31]]
y = train.iloc[ : , [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]]

x_val = val.iloc[ :,  [0, 1, 5, 6, 7, 19, 24, 25, 29, 31]]
y_val = val.iloc[ : , [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]]

x_pred = test.iloc[ :,  [0, 1, 5, 6, 7, 19, 24, 25, 29, 31]]
y_pred = test.iloc[ : , [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 30, 32, 33, 34]]

print(x.shape)  # (2895, 10)
print(y.shape)  # (2895, 25)

y = y.values
y_val = y_val.values
y_pred = y_pred.values

# x = np.array(x)
# y = np.array(y)


# x_train, x_test, y_train, y_test = train_test_split(
# x,y, train_size = 0.8, shuffle=True, random_state = 25)


# print(x_train.shape)  # (2316, 10)
# print(y_train.shape)  # (2316, 25)


print(y_pred.shape[1])

for i in range(y.shape[1]):
    
    
    # y_col = y_pred.iloc[: , i]
    
    y_train = y[ : , i]
    y_predict = y_pred[360 : , i]
    
    # size = int(len(y_train) * 0.5)
    train = y_train
    test =  y_predict
    history = [x for x in train]
    predictions = list()

    for t in range(len(test)):
	    model = ARIMA(history, order=(3,0,3) )
	    model_fit = model.fit(disp=1, maxiter=1000,solver='cg', method="mle")
	    output = model_fit.forecast()
	    yhat = output[0]
	    predictions.append(yhat)
	    obs = test[t]
	    history.append(obs)
	    print('predicted=%f, expected=%f' % (yhat, obs))
     
    
    
      
 


        

# y_pred = model.predict(x_pred)

predict = pd.DataFrame(predictions)
# predict = predict.iloc[360: ]
predict.to_csv('.\AI_2020\predict_ARIMA.csv', header=0, index=0)
