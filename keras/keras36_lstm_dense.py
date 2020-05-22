from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 함수형 모델로 리뉴얼 

#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11], [10, 11, 12], 
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])  # (13,3)

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) #(13,)

x_predict = array([55,65,75])   #(3,)

x = x.reshape(13,3)
x_predict = x_predict.reshape(1,3)  

y= y.reshape(13,1)

# x = x.reshape(x.shape[0], x.shape[1], 1)
# x_predict = x_predict.reshape(1,3,1)





# 2. 모델 구성



model = Sequential()
model.add(Dense(5, activation='relu', input_shape = (3,)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

'''
model.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 5)                 20
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_4 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 6
=================================================================
Total params: 116
Trainable params: 116
Non-trainable params: 0
_________________________________________________________________


'''
# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, callbacks= [ealry_stopping])




y_predict = model.predict(x_predict)
print(y_predict)        
print(y_predict.shape)   # (1,1)


