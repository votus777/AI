
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from keras.metrics import mse


import joblib

from statsmodels.tsa.arima_model import ARIMA

 
series = pd.read_csv('./AI_2020/train.csv', header = 1, index_col = [0,1])

series = series['경인선']

series = series [:200]

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# error = mse(test, predictions)
# print('Test MSE: %.3f' % error)

# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()