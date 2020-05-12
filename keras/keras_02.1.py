from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,3,4,5,6,7,8,9,10,11])
x_test = np.array([11,22,33,44,55,66,77,88,99,100])
y_test = np.array([12,23,34,45,56,67,78,89,100,101])

model = Sequential()
model.add(Dense(5, input_dim =1 , activation='relu'))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data = (x_test, y_test))

loss, acc = model.evaluate(x_test, y_test, batch_size =1)

output = model.predict(x_test)
print("결과물 : \n", output)

print("loss : ", loss)
print("acc : ", acc)
