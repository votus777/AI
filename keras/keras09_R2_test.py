'''
# 과제 : R2를 음수가 아닌 0.5이하로 줄이기   
# 레이어는 인풋, 아웃풋 포함 5개 이상, 노드는 레이어당 각각 5개 이상
# batch size = 1
# epochs = 100 이상 
# 데이터 수정 없음


R2 score :  0.4422005452443175
나오긴 나오는데 항상 나오진 않더라

대체 왜 이러는걸까?



'''

# 1. 데이터
import numpy as np


x = np.array(range(1,11))
y = np.array(range(1,11))

x_train = x[:1]
y_train = y[:1]


x_test = x[2:4]
y_test = y[2:4]


x_pred = np.array([16, 17, 18]) 




# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(2, input_dim = 1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(25))
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



#________RMSE 구하기___________________
from sklearn.metrics import mean_squared_error
def RMSE(y_test ,y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

# y_test = 실제값, y_pred = 예측값

print("RMSE : ", RMSE(y_test, y_predict))    




#________R2 구하기_____________________
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("R2 score : ", r2)


