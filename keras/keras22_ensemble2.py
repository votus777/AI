
# 2개 -> 3개

# 1. 데이터_________________________________
import numpy as np

x1 = np.array([range(1,101), range(311,411)]) #(100,2) X 2
x2 = np.array([range(101,201), range(411,511)]) 


y1 = np.array([range(711,811), range(711,811)]) #(100,2) X 3
y2 = np.array([range(501,601), range(711,811)])
y3 = np.array([range(411,511), range(611,711)]) 

###########################################
##############여기서 부터 수정##############
###########################################
x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
y3 = np.transpose(y3)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
   
    x1, y1, shuffle = False  , train_size = 0.8  #x_train = (80,3) ; x_test = (20,3)
)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
   
    x2, y2, shuffle = False  , train_size = 0.8  
)

from sklearn.model_selection import train_test_split
y3_train , y3_test = train_test_split(
   
    y3, shuffle = False  , train_size = 0.8  
)
'''
train_test_split 할 때는 굳이 쌍으로 넣지 않아도 된다!

'''


# 2. 모델 구성____________________________
from keras.models import Sequential, Model  # Model 추가
from keras.layers import Dense, Input # 마찬가지로 input layer를 추가



#model -------- 1
input1 = Input(shape=(2, ), name= 'input_1') 

dense1_1 = Dense(10, activation= 'relu', name= '1_1') (input1) 
dense1_2 = Dense(10,activation='relu', name = '1_2')(dense1_1)
dense1_3 = Dense(10,activation='relu', name = '1_3')(dense1_2)


#model -------- 2
input2 = Input(shape=(2, ), name = 'input_2') 

dense2_1 = Dense(10, activation= 'relu', name = '2_1')(input1) 
dense2_2 = Dense(10,activation='relu', name = '2_2')(dense2_1)
dense2_3 = Dense(10,activation='relu', name = '2_3')(dense2_2)

 


#이제 두 개의 모델을 엮어서 명시 
from keras.layers.merge import concatenate    #concatenate : 사슬 같이 잇다
merge1 = concatenate([dense1_3, dense2_3], name = 'merge') #파이썬에서 2개 이상은 무조건 list []

middle1 = Dense(10)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

################# output 모델 구성 ####################

output1 = Dense  (10,name = 'output_1')(middle1)
output1_2 = Dense (10, name = 'output_1_2')(output1)
output1_3 = Dense (10, name = 'output_1_3')(output1_2)
output1_4 = Dense (2, name = 'output_1_4')(output1_3)



output2 = Dense  (10,name = 'output_2')(middle1)
output2_2 = Dense (10,name = 'output_2_2')(output2)
output2_3 = Dense (10,name = 'output_2_3')(output2_2)
output2_4 = Dense (2, name = 'output_2_4')(output2_3)

output3 = Dense (10,name = 'output_3')(middle1)
output3_2 = Dense(10,name = 'output_3_2')(output3)
output3_3 = Dense(10,name = 'output_3_3')(output3_2)
output3_4 = Dense (2, name = 'output_3_4')(output3_3)

model = Model (inputs = [input1, input2], outputs= ([output1_4, output2_4, output3_4]))


# model.summary()

# 3. 훈이의 성실한 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], 
          [y1_train, y2_train, y3_train], epochs=50, batch_size = 1, validation_split= 0.25, verbose = 1)  #2개 이상은 모두 []로 묶어준다


#4. 평가, 예측____________________________________________
loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test,y3_test], batch_size = 1 ) # 여기도 역시 묶어준다

print(loss)



y1_predict, y2_predict, y3_predict = model.predict([x1_test,x2_test]) # 리스트의 함정에 조심!!!




#________RMSE 구하기_________________________________________
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ", (RMSE(y1_test, y1_predict) + RMSE(y2_test, y2_predict) + RMSE(y3_test, y3_predict))/3)



#________R2 구하기_____________________
from sklearn.metrics import r2_score

def R2(y_test, y_predict) :
    return r2_score(y_test, y_predict)

print("R2 score : ", (R2(y1_test,y1_predict)+R2(y2_test,y2_predict)+ R2(y3_test,y3_predict))/3)
# _____________________________________

# 여기까지가 대략 책 156p 까지 

