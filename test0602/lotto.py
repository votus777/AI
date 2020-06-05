import numpy as np

x = np.array(range(1,46)) 
y = np.array(range(1,46)) 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, test_size = 0.13, random_state = 3
)

# 2. 모델 구성____________________________
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(1))


# 3. 훈훈한 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size = 1, validation_split= 0.5)



#4. 평가, 예측____________________________________________
loss,mse = model.evaluate(x_test, y_test, batch_size = 1) 
print("loss : ", loss)
print("mse : ", mse) 



y_predict = model.predict(x_test)
print((np.around(y_predict)))