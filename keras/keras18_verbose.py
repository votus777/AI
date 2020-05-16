
'''

Verbose : '말 수가 많은'

말그대로 얘가 말이 많아진다. 안 물어본 것도 다 떠들어댄다는 소리

자세한 출력(학습상황)을 보여준다.



...고 알았는데 기본값이 말이 많은 거였고

조정할 수록 얘가 말이 짧아진다

'''


# 1. 데이터_________________________________
import numpy as np

x = np.array([range(1,101), range(311,411), range(100)])
y = np.array(range(711,811))


x = np.transpose(x)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = False  , train_size = 0.8  
)




# 2. 모델 구성____________________________
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 3)) # input 데이터가 3열 
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(1))                # 반면 output은 1열 


# 3. 훈제 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split= 0.5, verbose = 0) #verbose 추가 

'''

verbose = 0  -> 갑자기 짠하고 결과값 출력

verbose = 1  -> 평상시 보던 대로 나옴

verbose = 2  -> 프로그래스바 MIA 

verbose = 3  -> 2보다 더 간략

verbose = 4  -> epoch 숫자만

4 이후는 달라지는 것 없음

굳이 잘 보지도 않는거 일일히 출력하기보다 그냥 verbose = 0 으로 해버리면 좀 더 빠르게 결과를 볼 수 있다. 


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
