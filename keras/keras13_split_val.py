
# 1. 데이터_________________________________
import numpy as np

x = np.array(range(1,101)) # 1~100
y = np.array(range(101,201)) # w = 1, b = 100


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state =99, shuffle= True, test_size = 0.2 #train 80% // test 20% 
)

# random_state => random 난수, 숫자는 랜덤이지만 랜덤 난수66으로 고정된 랜덤이므로 다시 실행해도 숫자는 변하지 않는다 


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


# 3. 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split= 0.5)

'''
 validation data (x_val,y_val) 대신에 validation_split 파라미터를 넣어주었다. 
 그래서 따로 validarion data를 명시해주지 않아도 된다.

'''

#4. 평가, 예측____________________________________________
loss,mse = model.evaluate(x_test, y_test, batch_size = 1) 
print("loss : ", loss)
print("mse : ", mse) 



y_predict = model.predict(x_test)
print(y_predict)


#________RMSE 구하기_________________________________________
from sklearn.metrics import mean_squared_error
def RMSE(y_test ,y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

# y_test = 실제값, y_pred = 예측값

print("RMSE : ", RMSE(y_test, y_predict))    




#________R2 구하기_____________________
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("R2 score : ", r2)
# _____________________________________

