
# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])


x_pred = np.array([16, 17, 18]) 




# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size = 1)




#  4. 평가, 예측
loss,mse = model.evaluate(x_test, y_test, batch_size = 1) 
print("loss : ", loss)
print("mse : ", mse) 


'''
y_pred = model.predict(x_pred) 
print("y_pred : ", y_pred)
'''

y_predict = model.predict(x_test)
print(y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test ,y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

# y_test = 실제값, y_pred = 예측값

print("RMSE : ", RMSE(y_test, y_predict))    


'''
loss :  5.820766091346741e-12
mse :  5.8207661780829145e-12
[[11.000001]
 [12.000002]
 [13.000003]
 [14.000005]
 [15.      ]]
RMSE :  2.6634674155748254e-06
'''