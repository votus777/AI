

from keras.datasets import imdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. 데이터 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)




# y의 카테고리 갯수 출력
category = np.max(y_train) +1 
print('카테고리 : ', category)   # 카테고리 :  2




# 유니크한 y 값들 출력
y_dist = np.unique(y_train)
# print(y_dist)



# y_train_pd = pd.DataFrame(y_train)
# bbb = y_train_pd.groupby(0)[0].count()   # 주간 과제 : groupby 사용법 숙지 



# Pad Sequence
from keras.preprocessing.sequence import pad_sequences 
from keras.utils.np_utils import to_categorical 

x_train = pad_sequences(x_train, maxlen=500, padding='pre', value=0)  
x_test = pad_sequences(x_test, maxlen=500, padding='pre', value=0)   


# print(len(x_train[0]))   # 500
# print(len(x_train[-1]))  # 500


# print(x_train.shape)  (25000, 500)
# print(x_test.shape)   (25000, 500)

# 2. 모델

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten

from keras.layers import Bidirectional

# 꼭 시계열 데이터에서 미래만 예측(정항향) 하는게 아니라 과거(역방향)도 예측할 수 있다
# 특히 NLP에서는 더더욱 
# 앞 뒤 둘 다 테스트 함으로써  더 많은 데이터를 쓰는 방법

model = Sequential()

model.add(Embedding(1000, 128, input_length = 500))   
# model.add(LSTM(128))
model.add(Bidirectional(LSTM(128)))   # 양방향으로 연산하도록 만들겠다 
model.add(Dense(1, activation='sigmoid'))


model.summary()

# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 500, 128)          128000
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 128)               131584
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 129
# =================================================================
# Total params: 259,713
# Trainable params: 259,713
# Non-trainable params: 0
# _________________________________________________________________


# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 500, 128)          128000
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, 256)               263168   -> X2 배가 되었다 
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 257
# =================================================================
# Total params: 391,425
# Trainable params: 391,425
# Non-trainable params: 0
# _________________________________________________________________

'''

model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs = 10, validation_split= 0.2)

acc = model.evaluate(x_test,y_test)[1] 

print( 'acc : ' , acc)


y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet loss')
plt.plot(y_loss, marker = '.', c = 'blue', label = 'TrainSet loss')

plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

'''