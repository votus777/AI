
# keras49_classify.py


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

import keras.utils 

# 1. 데이터
x = np.array(range(1, 11))    # (10,)
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]) # (10,)



y= keras.utils.to_categorical(y)

'''
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


'''


# 다섯 개의 모델 



# 2. 모델 구성


model = Sequential()
model.add(Dense(10, activation='relu', input_dim = 1))
model.add(Dense(50, activation= 'softmax')) 
model.add(Dense(6, activation= 'softmax')) 



# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 50, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.fit(x,y, epochs= 8000, batch_size= 1, callbacks= [early_stopping])



# 평가 및 예측 




loss, acc = model.evaluate(x,y, batch_size=1)


x_predict = np.array([1,2,3,4,5])
y_predict = model.predict(x_predict)  

print('y_predict : ', y_predict)  
print('loss :', loss)
print('accuracy : ', acc)


'''
y_predict :  [[6.94022398e-04 // 9.37867224e-01 // 3.87556627e-02 // 1.02730263e-02 // 7.56665226e-03 // 4.84342873e-03]
 [5.68145770e-04 4.03201059e-02 6.93634033e-01 2.54882842e-01 6.73184264e-03 3.86301149e-03]
 [7.91960338e-04 1.04933217e-01 4.54412431e-01 3.38206112e-01 5.81025779e-02 4.35536690e-02]
 [4.92364750e-04 1.31446764e-01 1.21165805e-01 1.97330266e-01 2.73576766e-01 2.75988072e-01]
 [4.89754661e-04 1.31165043e-01 1.20062709e-01 1.96227774e-01 2.74630785e-01 2.77423918e-01]]   -> ....???



loss : 1.2446512058377266
accuracy :  0.4000000059604645





'''