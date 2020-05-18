
# 2개 -> 1개 

# 1. 데이터_________________________________
import numpy as np

x1 = np.array([range(1,101), range(311,411),range(411,511)]) 
x2 = np.array([range(101,201), range(411,511),range(511,611)]) 


y1 = np.array([range(101,201), range(411,511),range(100)]) 


###########################################
##############여기서 부터 수정##############
###########################################
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)




from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
   
    x1, y1, shuffle = False  , train_size = 0.8 
)

from sklearn.model_selection import train_test_split
x2_train, x2_test, = train_test_split(
   
    x2,  shuffle = False  , train_size = 0.8  
)




# 2. 모델 구성____________________________
from keras.models import Sequential, Model  
from keras.layers import Dense, Input 



#model -------- 1
input1 = Input(shape=(3, ), name= 'input_1') 

dense1_1 = Dense(8, activation= 'relu', name= '1_1') (input1) 
dense1_2 = Dense(18,activation='relu', name = '1_2')(dense1_1)
dense1_3 = Dense(4,activation='relu', name = '1_3')(dense1_2)


#model -------- 2
input2 = Input(shape=(3, ), name = 'input_2') 

dense2_1 = Dense(8, activation= 'relu', name = '2_1')(input1) 
dense2_2 = Dense(18,activation='relu', name = '2_2')(dense2_1)
dense2_3 = Dense(4,activation='relu', name = '2_3')(dense2_2)

 


#이제 두 개의 모델을 엮어서 명시 
from keras.layers.merge import concatenate    #concatenate : 사슬 같이 잇다
merge1 = concatenate([dense1_3, dense2_3], name = 'merge') #파이썬에서 2개 이상은 무조건 list []

middle1 = Dense(18)(merge1)
middle1 = Dense(18)(middle1)
middle1 = Dense(24)(middle1)
middle1 = Dense(10)(middle1)

################# output 모델 구성 ####################

output1 = Dense  (8,name = 'output_1')(middle1)
output1_2 = Dense (18, name = 'output_1_2')(output1)
output1_3 = Dense (8, name = 'output_1_3')(output1_2)
output1_4 = Dense (3, name = 'output_1_4')(output1_3)



model = Model (inputs = [input1, input2], outputs= (output1_4))



# 3. 훈이의 성실한 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], 
          y1_train, epochs=50, batch_size = 1, validation_split= 0.25, verbose = 1)  #2개 이상은 모두 []로 묶어준다


#4. 평가, 예측____________________________________________
loss = model.evaluate([x1_test,x2_test], y1_test, batch_size = 1 ) # 여기도 역시 묶어준다

print(loss)



y1_predict = model.predict([x1_test,x2_test]) # 리스트의 함정에 조심!!!




#________RMSE 구하기_________________________________________
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ", (RMSE(y1_test, y1_predict)))



#________R2 구하기_____________________
from sklearn.metrics import r2_score

def R2(y_test, y_predict) :
    return r2_score(y_test, y_predict)

print("R2 score : ", (R2(y1_test,y1_predict)))
# _____________________________________
