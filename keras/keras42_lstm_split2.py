import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))
size = 5                # timesteps = 4



def split_x(seq, size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size)]
        aaa.append([item for item in subset])
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)      # (96,5)


x = dataset[:, 0:4]     
y = dataset[:, 4]

x_predict = dataset[89: , 0:4]


x = np.reshape(x, (96,4,1))


# 1. train, test 분리할 것 (8:2)
# 2. 마지막 6개의 행을 predict로 만들고 싶다
# 3. validation을 넣을 것 (train의 20%)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    # x,y, random_state=99, shuffle=True,
    x,y, train_size=0.8
)  

#2. 모델

model = Sequential()
model.add(LSTM(5, input_shape=(4,1)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])




#4. 평가, 예측
loss, mse = model.evaluate(x,y, batch_size=1)
y_predict = model.predict(x)

print('loss: ', loss)
print('mse: ', mse)
print('y_predict: ', y_predict)
