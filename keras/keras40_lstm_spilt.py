
# kera40_lstm_split.py

# LSTM  모델을 완성하시오 

import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터 

a = np.array (range(1,11))
size = 5   # time_steps = 4


def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
   
    return np.array(aaa)


dataset = split_x(a,size) #(6,5)
print(type(dataset)) # <class 'numpy.ndarray'> -> split_x 함수의 return 값이 np.array()이기 떄문 

x =dataset [ :, 0:4]     # [ : ]  -> 모든 행을 가져오겠다. // [ :, 0:4] -> 모든 행의 각각 4번째 열까지 가져오겠다
y =dataset [ :, 4]

print(x.shape) #(6,4)
print(y.shape) #(6,)

x = x.reshape(6,4,1)
y = y.reshape(6,1)

#  x = np.reshape(x, (6,4,1))도 가능 

'''
1 2 3 4 5 6 7 8 9 10
    X     Y              [ : ]
1 2 3 4 | 5
2 3 4 5 | 6
3 4 5 6 | 7
4 5 6 7 | 8
5 6 7 8 | 9
6 7 8 9 | 10

'''
'''
dataset = 
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]   
 
 for 문을 구성해야 할 듯 싶다. 여기서 그냥  x = dataset [ :4] 해버리면 dataset list안의 4개가 튀어 나온다. 
각 리스트를 뽑아낸 뒤 그것을 [1,2,3,4], [5] 로 슬라이싱 하는 작업 ...

할 필요 없이 그냥
x = dataset [ :, 0:4]
y = dataset [ :, 4]

이렇게 하면 된다

겁나 해맸네..

'''




# 모델 구성 

model = Sequential()
model.add(LSTM(10, activation= 'relu', input_shape = (4,1)))
model.add(Dense(12)) 
model.add(Dense(12)) 
model.add(Dense(12))
model.add(Dense(10)) 
model.add(Dense(10)) 
model.add(Dense(1)) 



from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 50, mode ='auto')

model.compile(loss = 'mse', optimizer='adam', metrics = ['mse'])
model.fit(x,y, epochs= 5000, batch_size= 1, callbacks= [early_stopping])


loss, mse = model.evaluate(x,y, batch_size=1)
'''
#  metrics = ['mse'] 이거 뺴먹으면 
    'cannot unpack non-iterable float object' error 난다 

'''
x_predict = np.array ([11,12,13,14])
x_predict = np.reshape(x_predict, (1,4,1))

y_predict = model.predict(x_predict)  

print(y_predict)

print('loss :', loss)
print('mse : ', mse)

