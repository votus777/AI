

# 1개 -> 2개  


# 1. 데이터_________________________________
import numpy as np

x1 = np.array([range(1,101), range(301,401)]).T

y1 = np.array([range(711,811), range(711,811)]) 
y2 = np.array([range(101,201), range(411,511)]) 



###########################################
##############여기서 부터 수정##############
###########################################


y1 = np.transpose(y1)
y2 = np.transpose(y2)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
   
    x1, y1, y2, shuffle = False  , train_size = 0.8 
)




# 2. 모델 구성____________________________
from keras.models import Sequential, Model  
from keras.layers import Dense, Input 



#model -------- 1
input1 = Input(shape=(2, ), name= 'input_1') 

dense1_1 = Dense(8, activation= 'relu', name= '1_1') (input1) 
dense1_2 = Dense(20,activation='relu', name = '1_2')(dense1_1)
dense1_3 = Dense(8,activation='relu', name = '1_3')(dense1_2)


################# output 모델 구성 ####################

output1 = Dense  (40,name = 'output_1')(dense1_3)
output1_2 = Dense (40, name = 'output_1_2')(output1)
output1_3 = Dense (40, name = 'output_1_3')(output1_2)
output1_4 = Dense (2, name = 'output_1_4')(output1_3)

output2 = Dense  (40,name = 'output_2')(dense1_3)
output2_2 = Dense (40, name = 'output_2_2')(output1)
output2_3 = Dense (40, name = 'output_2_3')(output1_2)
output2_4 = Dense (2, name = 'output_2_4')(output1_3)



model = Model (inputs = input1, outputs= ([output1_4,output2_4]))



# 3. 훈이의 성실한 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train, 
          [y1_train, y2_train], epochs=70, batch_size = 1, validation_split= 0.25, verbose = 1)  #2개 이상은 모두 []로 묶어준다


#4. 평가, 예측____________________________________________
loss = model.evaluate(x1_test,[y1_test, y2_test], batch_size = 1 ) # 여기도 역시 묶어준다

print(loss)



y1_predict,y2_predict = model.predict(x1_test) # 리스트의 함정에 조심!!!




#________RMSE 구하기_________________________________________
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ", ((RMSE(y1_test, y1_predict)+RMSE(y2_test, y2_predict))/2))



#________R2 구하기_____________________
from sklearn.metrics import r2_score

def R2(y_test, y_predict) :
    return r2_score(y_test, y_predict)

print("R2 score : ", ((R2(y1_test,y1_predict)+R2(y2_test,y2_predict))/2))
# _____________________________________
