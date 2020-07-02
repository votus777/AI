
# 1. 데이터 

import numpy as np

x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

# 2. 모델 구성

from keras.models import Sequential
from keras.layers import Dense

model  = Sequential()

model.add(Dense(10, input_dim = 1, activation='relu'))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1))


from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam

optimizer1 = Adam(lr= 0.01)     #   9.791796037461609e-05    [1.5029156]
optimizer2 = RMSprop(lr= 0.01)  #   0.05919376760721207      [1.6568221]
optimizer3 = SGD(lr= 0.01)      #   0.00012463207531254739   [1.5135243]
optimizer4 = Nadam(lr= 0.01)    #   1.2364585018076468e-06   [1.4986882]
optimizer5 = Adadelta(lr= 0.01)    #   5.018414497375488    [0.28022382]   얘는 lr= 0.8 정도로 크게 줘야 된다 1.9917110876122024e-06,[1.4988055]


model.compile(loss = 'mse', optimizer=optimizer1, metrics=['mse'])

model.fit(x,y, epochs = 100)

loss = model.evaluate(x,y)

pred1 = model.predict([1.5, 2.5, 3.5, 4.5])


print("loss : ", loss)
print("pred1 : ", pred1)

fhhgorortmg,lmdlepoerpi