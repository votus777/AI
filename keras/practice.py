

import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, LSTM 



# 1. 데이터 
a = np.array(range(1,11))

size =5 

def split_x (seq,size) :
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[ i : (i+ size)] 
        aaa.append([item for item in subset])

    return np.array(aaa)



data_set = split_x(a,size)

x = data_set [ : , 0:4]
y = data_set [ : , 4]


# x = x.reshape(6,4,1)


# print(x)
# print(y)


# 2. 모델 

model = Sequential()
model.add (Dense(10, input_shape = (4,))) 
model.add(Dense(5))
model.add(Dense(5)) 
model.add(Dense(5)) 
model.add(Dense(5)) 
model.add(Dense(1)) 




# 3. 학습 

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 50, mode ='auto')

model.compile(loss = 'mse', optimizer='adam', metrics = ['mse'])
model.fit(x,y, epochs= 5000, batch_size= 1, callbacks= [early_stopping])


loss, mse = model.evaluate(x,y, batch_size=1)
'''
#  metrics = ['mse'] 이거 뺴먹으면 
    'cannot unpack non-iterable float object' error 난다 

'''
x_predict = np.array ([[11,12,13,14]])
# x_predict = np.reshape(x_predict, (1,4,1))

y_predict = model.predict(x_predict)  

print(y_predict)

print('loss :', loss)
print('mse : ', mse)








