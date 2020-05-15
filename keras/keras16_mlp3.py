#from keras14


'''

하나에서 세개의 아웃풋 뽑아내기 


'''

# 1. 데이터_________________________________
import numpy as np

x = np.array(range(1,101))
y = np.array([range(101,201), range(711,811), range(100)]).T




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = False  , train_size = 0.8  
)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state =99, shuffle= False, test_size = 0.2 #train 80% // test 20% 
)



# 2. 모델 구성____________________________
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 1))   #input 1개
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(3))     # output은 3개

'''

keras14.py 에서 썼던 파라미터를 그대로 가져다 썼더니 지표들이 난리가 난다!

다시 조정해주자 

왜그럴까? 

1개에서 3개를 뽑는 것보다 3개에서 1개를 뽑는게 더 자주 쓰이는 것과 연관있을듯 

'''

# 3. 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split= 0.5) #validation 40%



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

