
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.datasets  import load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt 
import numpy as np


diabetes = load_diabetes()
x, y = diabetes.data, diabetes.target


# print(x.shape)   # (442, 10)
# print(y.shape)   # (442, )


x = x[:, np.newaxis, 3]

# print(x[0])  # [[0.03807591 0.05068012 0.06169621 0.02187235]]
# print(x.shape)  # (442, 1, 4)

print(y[0]) 
print(x.shape)   # (442, 1)

y = y.reshape(442,1)

#####################################
standard_scaler = StandardScaler()    

x = standard_scaler.fit_transform(x)


minmax_scaler = MinMaxScaler()

x = minmax_scaler.fit_transform(x)

y = minmax_scaler.fit_transform(y)


####################################



x = x.reshape(442,1,1)*100





from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x,y, shuffle = True  , train_size = 0.8  
)






# 모델  

model = Sequential()
model.add(LSTM(16, activation='relu', input_shape = (1,1)))
model.add(Dense(16, activation= 'relu')) 
model.add(Dense(32, activation= 'relu' )) 
model.add(Dropout(0.2))


model.add(Dense(64, activation= 'relu')) 
model.add(Dense(64, activation= 'relu')) 
model.add(Dense(64, activation= 'relu')) 
model.add(Dropout(0.2))


model.add(Dense(32, activation= 'relu')) 
model.add(Dense(16, activation= 'relu')) 
model.add(Dense(16, activation= 'relu')) 
model.add(Dropout(0.2))

model.add(Dense(1, activation= 'relu')) 


# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 10, mode ='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

hist = model.fit(x_train,y_train, epochs= 10000, batch_size= 2, validation_split= 0.2,  callbacks= [early_stopping])



# 평가 및 예측 


loss, mse = model.evaluate(x_test,y_test, batch_size=1)


print('loss :', loss)
print('mse : ', mse)


