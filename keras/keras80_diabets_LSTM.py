
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.datasets  import load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import matplotlib.pyplot as plt 
import numpy as np


diabetes = load_diabetes()
x, y = diabetes.data, diabetes.target


# print(x.shape)   # (442, 10)
# print(y.shape)   # (442, )
print(x[0])
print("===================")


x = x[:, 0:4]

print(x[0])  # [[0.03807591 0.05068012 0.06169621 0.02187235]]

x = np.round(x,4)

print(x[0])     # [0.0381 0.0507 0.0617 0.0219]
print(y[0]) 
print(x.shape)   # (442, 4)


#####################################

standard_scaler = StandardScaler()    

# x = standard_scaler.fit_transform(x)


robust_scaler = RobustScaler()

x = robust_scaler.fit_transform(x)

####################################

print(x.shape)

x = x.reshape(442,4,1)*100



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)




# 모델  

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape = (4,1)))
model.add(Dense(16, activation= 'relu')) 
model.add(Dense(32, activation= 'relu' )) 
model.add(Dropout(0.2))


model.add(Dense(64, activation= 'relu')) 
model.add(Dense(64, activation= 'relu')) 
model.add(Dense(64, activation= 'relu')) 
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
from keras.optimizers import Adam

early_stopping = EarlyStopping( monitor='loss', patience= 10, mode ='auto')

model.compile(loss='mse', optimizer= Adam (lr=0.001, beta_1=0.9, beta_2=0.999), metrics=['mse'])

hist = model.fit(x_train,y_train, epochs= 1000, batch_size= 2, validation_split= 0.2)



# 평가 및 예측 


loss, mse = model.evaluate(x_test,y_test, batch_size=1)


print('loss :', loss)
print('mse : ', mse)


y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)

print("R2 score : ", r2)


'''

loss : 3880.6124057280886
mse :  3880.61279296875
R2 score :  0.40212287079390385


'''