
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn  import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 
import numpy as np

breast_cancer = datasets.load_breast_cancer()


x = breast_cancer.data
y = breast_cancer.target

print(x.shape)   #(569, 30)  # 소수점의 향연 
print(y.shape)   #(569,)    # 1과 0의 향연


x = x.reshape(569,1,5,6)


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)



# 모델  

model = Sequential()
model.add(Conv2D(12,(1,1), activation='relu', input_shape = (1,5,6)))
model.add(Dropout(0.2))

model.add(Conv2D(12,(1,1), activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation= 'sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(32, activation= 'sigmoid'))
model.add(Dense(10, activation= 'sigmoid'))

model.add(Flatten())


model.add(Dense(10, activation= 'sigmoid'))

model.add(Dense(1, activation= 'sigmoid'))




# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 30, mode ='auto')

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

hist = model.fit(x_train,y_train, epochs= 10000, batch_size= 1, validation_split= 0.2,  callbacks= [early_stopping])



# 평가 및 예측 



loss, acc = model.evaluate(x_test,y_test, batch_size=1)
  

print('loss :', loss)
print('accuracy : ', acc)

'''

loss : 0.18060839686209842
accuracy :  0.9210526347160339

'''