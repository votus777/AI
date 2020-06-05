
# 1. 데이터_________________________________
import numpy as np

x = np.array(range(1,101)) # 1~100
y = np.array(range(101,201)) # w = 1, b = 100


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state =99, shuffle= True, test_size = 0.4 #train 60% // test 40% 
)

# random_state => random 난수, 숫자는 랜덤이지만 랜덤 난수66으로 고정된 랜덤이므로 다시 실행해도 숫자는 변하지 않는다 
# 여담으로 sklearn 사이트에 따르면 가장 인기있는   random_state 숫자는 0 과 42라고 한다.  42...어디서 많이 들어본 숫자인데..( 영화 '은하수를 여행하는 히치하이커를 위한 안내서' 참고)

x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, random_state=99, shuffle= True, test_size=0.5)  #test 20% 중 -> 50% test // 50% validation 

# train : validation : test = 6 : 2 : 2

'''
훈련한 범위 밖의 예측은 틀릴 확률이 매우 높다. 미래 예측이 힘든 이유. 
그래서 shuffle 이 default로 있다. 그래서 섞어서 최대한 범위를 넓게 잡는다. 

여기서 shuffle의 조건 
- X,Y가 서로 섞이지는 않는다. X는 X 안에서, Y는 Y안에서만 섞인다. 



'''

'''
print(x_test,x_train,x_val)
print(y_test,y_train,y_val)



'''
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


# 3. 훈훈하게 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_data=(x_val,y_val))



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

