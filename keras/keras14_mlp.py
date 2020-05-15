#네

#Multi Layer Perceptron (MLP) 
#요새 MLB보다 KBO가 더 핫하다더라 

# 1. 데이터_________________________________
import numpy as np

x = np.array([range(1,101), range(311,411), range(100)])
y = np.array([range(101,201), range(711,811), range(100)])



print(x.shape)

x = np.transpose(x)   # (100,3)
y = np.transpose(y)



'''
print(x.shape)

(   1   2   3   4   5   ~    99 100 

  311 312 313 314 315   ~   409 410 

   0   1   2   3   4   ~   98  99  )

(3,100) 3행 100열  //  열 = 컬럼(column)

    [1 ~ 100]
    [311 ~ 410]
    [0 ~ 100]


'''
'''

       그런데       중요!!                ※열 우선, 행 무시※

                                 행은 어차피 데이터 개수에 따라 달라진다
                        그보다 열에 따라서 input_dim이 정해지기에 더 중요하다.
                         사실, 100행 3열이나 3행 100열이나 모델은 잘 돌아간다.
                          문제는 새로운 열(데이터)을 추가할 때 성가시다는 점이다. 
      
                        어차피 나중에 shape 값 넣어줄 떄 열의 갯수만 입력하게 된다. 


'''


'''

그래서 우리는 이것은 반대로 바꿔야 한다. input_dim=3이 되도록  






'''


'''
print(x.T)

[  1   2   3]
[  4   5   6]
[  7   8   9]
[ 10  11  12]
[ 13  14  15]
[ 16  17  18]
[ 19  20  21]
[ 22  23  24]
[ 25  26  27]
[ 28  29  30]

~

'''




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = False  , train_size = 0.8  
)

# print(x_train)
# print(x_val)
# print(x_test)





from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state =99, shuffle= False, test_size = 0.2 #train 80% // test 20% 
)



# 2. 모델 구성____________________________
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 3))   #데이터가 바뀌었으니 조정해준다
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(3))                   #마찬가지


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

