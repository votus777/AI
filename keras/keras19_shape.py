


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

# model.add(Dense(5, input_dim = 3)) 
model.add(Dense(5, input_shape = (3, ))) #행무시
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(1))                


# 3. 돌고 도는 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split= 0.5, verbose = 3) #verbose 추가 



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



print("RMSE : ", RMSE(y_test, y_predict))    




#________R2 구하기_____________________
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("R2 score : ", r2)
# _____________________________________
