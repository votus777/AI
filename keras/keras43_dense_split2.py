
# kera40_lstm_split.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))
size = 5               

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)


x = dataset[:94, 0:4]      
y = dataset[:94, 4]

x_predict = dataset[-6:, 0:4]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    # x,y, random_state=99, shuffle=True,
    x,y, train_size=0.8
)  



#2. 모델

model = Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=30, batch_size=1, verbose=1, callbacks=[early_stopping])


#4. 평가, 예측

loss, mse = model.evaluate(x_test,y_test, batch_size=1)
y_predict = model.predict(x_predict)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: ', y_predict)
