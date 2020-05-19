'''
 sequential model -> functional model

간단한 모델은 sequential model로 직관적이고 간단하게 만들고

다중 인풋 & 아웃풋 같은 복잡한 모델을 구성하기 위해서 함수형 모델을 이용할 수 있다. 


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
from keras.models import Sequential, Model  # Model 추가
from keras.layers import Dense, Input # 마찬가지로 input layer를 추가
'''
model = Sequential()
model.add(Dense(5, input_dim = 3))  
model.add(Dense(4))
model.add(Dense(1))                
'''

'''
함수형 모델은 처음부터 input 과 output을 명시해줘야 함

'''




input1 = Input(shape=(3, )) 
#변수명(소문자) 설정, Sequential에서는 layer 라고 그냥 해도 되었지만 functional 에서는 input을 정의

dense1 = Dense(5, activation= 'relu')(input1) # 역시 인풋 명시
dense1 = Dense(4,activation='relu')(dense1)
dense1 = Dense(3,activation='relu')(dense1)


output1 = Dense(1)(dense1) 

model = Model(inputs=input1, outputs = output1) # 요놈이 함수형 모델이다 명시, 범위가 어디인지(hidden layer제외) 

'''
so 이제부터 model을 호출하면 위에 놈이 튀어나오겠지 

그런데 keras.io 문서에 따르면 함수형 모델을 호출할 때는 모델 뿐만 아니라 가중치까지 재사용된다고 한다. (아직은 몰라도 될 것 같다)



'''


# 3. 훈수 두는 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split= 0.2, verbose = 1) 


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

'''
loss :  7.0033594965934755e-06
mse :  7.003359314694535e-06
[[791.0022 ]
 [792.00214]
 [793.0022 ]
 [794.00226]
 [795.0023 ]
 [796.00244]
 [797.0025 ]
 [798.00244]
 [799.0025 ]
 [800.0026 ]
 [801.0027 ]
 [802.00275]
 [803.00275]
 [804.00275]
 [805.0028 ]
 [806.0029 ]
 [807.003  ]
 [808.0031 ]
 [809.00305]
 [810.0031 ]]
RMSE :  0.002645541371484355
R2 score :  0.9999997895070933

'''