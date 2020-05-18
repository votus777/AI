
# 1. 데이터_________________________________
import numpy as np



x = np.array(range(1,101)) # 1~100
y = np.array(range(101,201)) # w = 1, b = 100


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = False  , train_size = 0.8  
)



x_test, x_val, y_test, y_val = train_test_split(
   
    x_test, y_test, shuffle = False , test_size=0.5
)  

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


# 3. 훈련_______________________________________________________________________________
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


loss :  2.8011389076709746e-06
mse :  2.8011388621962396e-06
[[181.00153]
 [182.00162]
 [183.00156]
 [184.00162]
 [185.00166]
 [186.00171]
 [187.00174]
 [188.00172]
 [189.00168]
 [190.00172]]
RMSE :  0.0016570721758366203
R2 score :  0.9999996671650672
