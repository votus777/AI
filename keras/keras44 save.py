 # kears 44 _ save.py

                   # 모델 저장하기 

import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, LSTM



# 2. 모델 구성 

model = Sequential()
model.add(LSTM(5, activation='relu', input_shape = (4,1)))  

model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))

'''
model.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 10)                480
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 10)                60
_________________________________________________________________
dense_3 (Dense)              (None, 10)                110
_________________________________________________________________
dense_4 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 6
=================================================================
Total params: 766
Trainable params: 766
Non-trainable params: 0
_________________________________________________________________
'''
# 모델 저장하기 

model.save(".//model//save_keras44.h5") 
# model.save(".\model\save_keras44.h5")  
# model.save("./model/save_keras44.h5")  
# 모두 가능 

print("저장이 잘되었는지 확인용")  # 만약 위에서 오류가 떴으면 출력이 안되겠지 

