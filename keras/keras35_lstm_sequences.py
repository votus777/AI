

from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM



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

dense1 = (LSTM(10, activation='relu', input_shape = (3,1), return_sequences=True))(input1)  
dense1_1 = (LSTM(10, activation='relu', input_shape = (3,1)))(dense1)  

'''

LSTM 레이어는 각 시퀀스에서 출력을 할 수도 있고 마지막 시퀀스에서 출력을 할 수도 있다. 

LSTM 레이어를 여러 개로 쌓아올릴 떄는 return_sequences 값을 True로 해줘서 각 시퀀스에서 값이 나오도록 한다. [h1,h2,h3,...] 


_________________

[1,2,3] 한 덩이가 첫번째 return_sequences=False LSTM layer에 들어가면 420번의 연산을 거치고 수정된 [1,2,3] 하나가 나오지만 

return_sequences=True 에서는 각 유닛들을 거쳐가며 나오는 값들을 모두 output으로 다름 LSTM에 전달해주게 된다. 
___________________


return_sequences를 안쓰면( False가 default ) lstm은 3차원으로 입력을 받아야하는데 2차원으로 받게되서 error
뜻은 원래 있던 차원 형식으로 전달해주겠다.  


마찬가지로 아래의 dense layer에 lstm layer의 output을 보낼떄 return_sequences를 true로 해버리면 error 
dense는 2차원 값을 받기 때문이다. 


Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 3, 1)         0
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 3, 1)         0
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 3, 10)        480         input_1[0][0] ------------> 여기서는 인풋이 3차원 1개로 들어왔는데 // g*h(h+i_1+1) = 480 // g=4, h=10, i_1= 1
                           
            return sequence = False면 output은 (None, 10) 이 나오는데 여기서는 (None, 3, 10)
                                        (batch_size, units)        (batch_size, time_steps, units)  -> time_steps 추가가....아니라

                                        제일 끝으로 가서 마지막에 feature 갯수로 추가된다! 실질적인 feature의 갯수가 10개로 증폭 
                                        그래서 증폭이 된 것을  return sequence = True로 해서 다음 LSTM이 받을 수 있는 것이다. 
                                                                          
                        

__________________________________________________________________________________________________
lstm_3 (LSTM)                   (None, 3, 10)        480         input_2[0][0]
__________________________________________________________________________________________________
lstm_2 (LSTM)                   (None, 10)           840         lstm_1[0][0]   -----------> input이 10개 들어왔다!           //g* h(h+i_2+1) = 840 // g=4, h=10, i_2 = 10
                                                                                             각각의 sequences에서 값을 받기 때문 -> RNN의 작동방식을 생각          
__________________________________________________________________________________________________
lstm_4 (LSTM)                   (None, 10)           840         lstm_3[0][0]
__________________________________________________________________________________________________
merge (Concatenate)             (None, 20)           0           lstm_2[0][0]
                                                                 lstm_4[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 10)           210         merge[0][0]
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 10)           110         dense_7[0][0]
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 10)           110         dense_8[0][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            11          dense_9[0][0]
==================================================================================================
Total params: 3,081
Trainable params: 3,081
Non-trainable params: 0
__________________________________________________________________________________________________




'''

dense1_2 =(Dense(5))(dense1_1)
dense1_3 =(Dense(12))(dense1_2)
dense1_4 =(Dense(12))(dense1_3)


# model 2___________________

input2 = Input(shape=(3,1)) 

dense2 = (LSTM(10, activation='relu', input_shape = (3,1), return_sequences=True))(input2)  
dense2_1 = (LSTM(10, activation='relu', input_shape = (3,1)))(dense2)  


dense2_2 =(Dense(5))(dense2_1)
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
