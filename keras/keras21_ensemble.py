


# 1. 데이터_________________________________
import numpy as np

x1 = np.array([range(1,101), range(311,411), range(100)])
y1 = np.array([range(711,811), range(711,811), range(100)])


x2= np.array([range(101,201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)




from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
   
    x1, y1, shuffle = False  , train_size = 0.8  #x_train = (80,3) ; x_test = (20,3)
)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
   
    x2, y2, shuffle = False  , train_size = 0.8  
)





# 2. 모델 구성____________________________
from keras.models import Sequential, Model  # Model 추가
from keras.layers import Dense, Input # 마찬가지로 input layer를 추가



#model -------- 1
input1 = Input(shape=(3, ), name= 'input_1') 

dense1_1 = Dense(5, activation= 'relu', name= '1_1') (input1) 
dense1_2 = Dense(4,activation='relu', name = '1_2')(dense1_1)
dense1_3 = Dense(3,activation='relu', name = '1_3')(dense1_2)


#model -------- 2
input2 = Input(shape=(3, ), name = 'input_2') 

dense2_1 = Dense(10, activation= 'relu', name = '2_1')(input1) 
dense2_2 = Dense(8,activation='relu', name = '2_2')(dense2_1)
dense2_3 = Dense(6,activation='relu', name = '2_3')(dense2_2)
dense2_4 = Dense(4,activation='relu', name = '2_4')(dense2_3)

  

#이제 두 개의 모델을 엮어서 명시 
from keras.layers.merge import concatenate    #concatenate : 사슬 같이 잇다
merge1 = concatenate([dense1_3, dense2_4], name = 'merge') #파이썬에서 2개 이상은 무조건 list []

middle1 = Dense(30)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

################# output 모델 구성 ####################

output1 = Dense (30)(middle1)
output1_2 = Dense (7)(output1)
output1_3 = Dense (3)(output1_2)


output2 = Dense (30)(middle1)
output2_2 = Dense (7)(output2)
output2_3 = Dense (3)(output2_2)

model = Model (inputs = [input1, input2], outputs= ([output1_3, output2_3]))

'''
model.summary()

Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 3)            0
__________________________________________________________________________________________________
2_1 (Dense)                     (None, 10)           40          input_1[0][0]
__________________________________________________________________________________________________
1_1 (Dense)                     (None, 5)            20          input_1[0][0]
__________________________________________________________________________________________________
2_2 (Dense)                     (None, 8)            88          2_1[0][0]
__________________________________________________________________________________________________
1_2 (Dense)                     (None, 4)            24          1_1[0][0]
__________________________________________________________________________________________________
2_3 (Dense)                     (None, 6)            54          2_2[0][0]
__________________________________________________________________________________________________
1_3 (Dense)                     (None, 3)            15          1_2[0][0]
__________________________________________________________________________________________________
2_4 (Dense)                     (None, 4)            28          2_3[0][0]
__________________________________________________________________________________________________
merge (Concatenate)             (None, 7)            0           1_3[0][0]
                                                                 2_4[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 30)           240         merge[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 5)            155         dense_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 7)            42          dense_2[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 30)           240         dense_3[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 30)           240         dense_3[0][0]
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 7)            217         dense_4[0][0]
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 7)            217         dense_7[0][0]
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 3)            24          dense_5[0][0]
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 3)            24          dense_8[0][0]
==================================================================================================
Total params: 1,668
Trainable params: 1,668
Non-trainable params: 0

'''




# 3. 훈민정음 훈련_______________________________________________________________________________
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], 
          [y1_train, y2_train], epochs=10, batch_size = 1, validation_split= 0.25, verbose = 1)  #2개 이상은 모두 []로 묶어준다


#4. 평가, 예측____________________________________________
loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test], batch_size = 1 ) # 여기도 역시 묶어준다

print(loss)

'''

#리턴하는 값의 개수보다 리턴 받을 변수의 값의 개수가 더 적으면 에러가 난다.

[16702.365771484376, 9404.228515625, 7298.1357421875, 9404.228515625, 7298.1357421875]
    전체 loss값         loss_1             loss_2         mse_1             mse_2

'''




y1_predict, y2_predict = model.predict([x1_test,x2_test]) # 리스트의 함정에 조심!!!




#________RMSE 구하기_________________________________________
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print("RMSE : ", (RMSE(y1_test, y1_predict) + RMSE(y2_test, y2_predict))/2)



#________R2 구하기_____________________
from sklearn.metrics import r2_score

def R2(y_test, y_predict) :
    return r2_score(y_test, y_predict)

print("R2 score : ", (R2(y1_test,y1_predict)+R2(y2_test,y2_predict))/2)
# _____________________________________


