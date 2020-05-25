
# kera40_lstm_split.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))
size = 5                # timesteps = 4

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)


x = dataset[:, 0:4]     
y = dataset[:, 4]

print(x)
print(y)

# x = np.reshape(x, (6,4,1))
# x = x.reshape(6,4,1)과 같은 표현

#2. 모델구성

model = Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Dense(5))
model.add(Dense(1))

#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=30, batch_size=1, verbose=1,
        callbacks=[early_stopping])


#4. 평가, 예측

loss, mse = model.evaluate(x,y, batch_size=1)
y_predict = model.predict(x)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: ', y_predict)


'''

import numpy as np 
from keras.models import Sequential , Model
from keras.layers import Dense, LSTM, Input

#1. 데이터 

a = np.array (range(1,11))
size = 5   # time_steps = 4


def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
   
    return np.array(aaa)


dataset = split_x(a,size)
# print(type(dataset))

x =dataset [:, 0:4]    
y =dataset [:, 4]

# print(x.shape) #(6,4)
# print(y.shape) #(6,)

# x = x.reshape(6,4,1)
# y = y.reshape(6,1,1)




#  x = np.reshape(x, (6,4,1))도 가능 



# 모델 구성 
input1 = Input(shape=(4,))  

dense1_2 =(Dense(5))(input1)
dense1_3 =(Dense(5))(dense1_2)
dense1_4 =(Dense(5))(dense1_3)
dense1_5 =(Dense(5))(dense1_4)

output1 = (Dense(1))(dense1_5)

model = Model(inputs=input1, outputs = output1)





from keras.callbacks import EarlyStopping  
early_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, batch_size = 1, callbacks= [early_stopping])



loss, mse = model.evaluate(x,y, batch_size=1)
y_predict = model.predict(x)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: ', y_predict)


'''