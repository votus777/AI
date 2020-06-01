import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.datasets import mnist

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint



# 데이터  

datasets = np.load('./data/iris_aaa.npy')

x = datasets[ : , : 4 ]
y = datasets[ : , np.newaxis, -1]


y= np_utils.to_categorical(y)


# print(x)   # [5.1 3.5 1.4 0.2]
# print(y)   #  [0. 0. 1.]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)


# 모델 

# 땡겨오자 keras76.py 
model = load_model('./model/sample/iris/model_iris.h5')


# 컴파일 및 훈련 


from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])
 
hist = model.fit(x_train,y_train, epochs= 10000, batch_size= 1, validation_split= 0.25,  callbacks= [early_stopping])


# 평가 

loss, acc = model.evaluate(x_test,y_test, batch_size=1)


print('loss :', loss)
print('accuracy : ', acc)


'''

loss : 0.0827335912414128
accuracy :  0.9333333373069763


'''