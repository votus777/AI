
# keras49_classify.py


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

import keras.utils 

# from keras.utils import np_utils

# 1. 데이터
x = np.array(range(1, 11))    # (10,)
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]) # (10,)



y= keras.utils.to_categorical(y)

y = y[ :, 1:]


'''



# One Hot encoding 
 
 y= keras.utils.to_categorical(y)   ( 혹은 sklearn 안에 있는 categorical() 을 사용하면 0 없이 잘 나온다. )

데이터를 이차원으로 바꿔줌

필수로 들어간다. 

print(y)

[[0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]]

(10,6) 

그런데 이걸 (10,5)로 만들 수 없을까 
거슬게 앞에 0이 계속 들어간다. 

이건 # 과제

y = y[ :, 1:]

[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]


print(y.shape) # (10,5)



'''




# 2. 모델 구성


model = Sequential()
model.add(Dense(10, activation='relu', input_dim = 1))
model.add(Dense(50, activation= 'softmax')) 
model.add(Dense(5, activation= 'softmax')) 



# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.fit(x,y, epochs= 10000, batch_size= 1, callbacks= [early_stopping])



# 평가 및 예측 




loss, acc = model.evaluate(x,y, batch_size=1)


x_predict = np.array([1,2,3,4,5])
y_predict = model.predict(x_predict)  

print('y_predict : ', np.around(y_predict))  
print('loss :', loss)
print('accuracy : ', acc)




'''
y_predict :  
[[2.6248332e-11 //9.9999988e-01// 1.6721158e-07 3.5981054e-10 7.7639883e-10 5.2086224e-09]
 [2.8180942e-11 1.0601286e-07 //9.9999464e-01// 5.1895831e-06 1.1149126e-12 1.1618150e-10]
 [2.2383173e-09 1.1898872e-08 4.6729227e-05 //9.9986660e-01// 8.6681968e-05 2.7839858e-08]
 [7.9115714e-10 7.8833734e-10 1.5837379e-08 1.2615275e-04 //9.9971086e-01// 1.6291614e-04]
 [4.3148929e-09 7.5874396e-04 3.1531947e-06 2.7333047e-07 6.8365771e-04 //9.9855417e-01//]]


잘 안보이지만 일단 수를 근사치로 확인해보자


9.9999988e-01 -> 0.99
2.6248332e-11 - > 0.00

[ 0, 1, 0, 0, 0, 0]
[ 0, 0, 1, 0, 0, 0]
[ 0, 0, 0, 1, 0, 0]
[ 0, 0, 0, 0, 1, 0]
[ 0, 0, 0, 0 ,0, 1]

이렇게 보면 값은 잘 분류가 된 것 같다. 



loss : 0.12215596875466801
accuracy :  1.0



_______________np.around(y_predict)을 이용하면 


[[0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]]
loss : 0.5687563135441394
accuracy :  0.6000000238418579

이렇게 한 번에 깔끔하게 나온다. 


_______________아래는  y = y[ :, 1:] 해준 후 결과

y_predict :  
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]

loss : 0.03297702865689871
accuracy :  1.0

'''