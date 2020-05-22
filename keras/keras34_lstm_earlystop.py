

from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

# 33번 파일에 이미 적용


#데이터 

x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11], [10, 11, 12], 
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])  

x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60], 
           [50, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
           [90, 100, 110], [100, 110, 120], 
           [2, 3, 4], [3, 4, 5], [4, 5, 6]])  

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) 

x1_predict = array([55, 65, 75])
x2_predict = array([65, 75, 85])


x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)


# 2. 모델 구성



# model 1____________________
input1 = Input(shape=(3,1)) 

dense1 = (LSTM(10, activation='relu', input_shape = (3,1)))(input1)  
dense1_1 =(Dense(12))(dense1)
dense1_2 =(Dense(12))(dense1_1)
dense1_3 =(Dense(12))(dense1_2)
dense1_4 =(Dense(12))(dense1_3)


# model 2___________________

input2 = Input(shape=(3,1)) 

dense2 = (LSTM(10, activation='relu', input_shape = (3,1)))(input2)  
dense2_1 =(Dense(12))(dense2)
dense2_2 =(Dense(12))(dense2_1)
dense2_3 =(Dense(12))(dense2_2)
dense2_4 =(Dense(12))(dense2_3)


# 모델 병합
from keras.layers.merge import concatenate   
merge1 = concatenate([dense1_1, dense2_1], name = 'merge') 

middle1_1 = Dense(10)(merge1)
middle1_2 = Dense(10)(middle1_1)

# 아웃풋 
output1_1 = Dense (10)(middle1_2)
output1_2 = Dense (1)(output1_1)

model = Model(inputs=[input1,input2], outputs = output1_2)

model.summary()


'''
# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 45,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit([x1,x2],y, epochs= 10000, callbacks= [ealry_stopping]) 



# 4. 평가, 예측

y_predict = model.predict([x1_predict,x2_predict])


print(y_predict)
print(y_predict.shape) #(1,1)


#  데이터가 워낙 이상해서 값이 이상하게 나온다 
'''