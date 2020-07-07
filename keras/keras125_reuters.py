
from keras.datasets import reuters
from keras.regularizers import l2, l1

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer 

# 1. 데이터 
(x_train, y_train), (x_test,y_test) = reuters.load_data(num_words = 2000, test_split = 0.2)

# print(x_train.shape, x_test.shape) (8982,) (2246,)
# print(y_train.shape, y_test.shape) (8982,) (2246,)  

# print(x_train[0])
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 2, 111, 16, 369, 186, 
#  90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 
#  7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 
#  11, 15, 7, 48, 9, 2, 2, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 
#  15, 16, 8, 197, 2, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]

# print(y_train[0]) # 3


# print(x_train[0].shape) -> error
print(len(x_train[0]))  #87    -> 문장별 길이가 일정하지 않다   -> pad sequence!




# y의 카테고리 갯수 출력
category = np.max(y_train) +1 
print('카테고리 : ', category)  
# 카테고리 :  46



# 유니크한 y 값들 출력
y_dist = np.unique(y_train)
# print(y_dist)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]


y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()   # 주간 과제 : groupby 사용법 숙지 

'''
y_train의 0 열의 값을 가지는 행들에서 [0]번쨰 data만 선택하여 갯수를 count  -> 여기서는 y가 ( n , )이기 때문에 (0)[0] = 전체선택 

print(bbb.shape)  #(46,)
print(bbb)
0       55
1      432
2       74
3     3159
.
.

'''




# Pad Sequence
from keras.preprocessing.sequence import pad_sequences 
from keras.utils.np_utils import to_categorical 

x_train = pad_sequences(x_train, maxlen=300, padding='post', value=0)   # truncating = 'pre'  -> maxlen 초과할 경우 앞부분 날리겠다, dafault
x_test = pad_sequences(x_test, maxlen=300, padding='post', value=0)   


# print(len(x_train[0]))  # 87 -> 100
# print(len(x_train[-1]))  # 105 -> 100

# One-Hot-Encoding 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(x_train.shape)  # (8982, 100)
print(x_test.shape)   # (2246, 100)

sin = tf.math.sin

# 2. 모델

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, BatchNormalization, Dropout
from keras.optimizers import Nadam

model = Sequential()

model.add(Embedding(1000, 500, input_length = 300))   
model.add(LSTM(128, activation = sin, kernel_regularizer= l1(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(46, activation='softmax'))

optimizer = Nadam(clipvalue=1.0)

# model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 100, 10)           10000
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               44400
# _________________________________________________________________
# dense_1 (Dense)              (None, 46)                4646
# =================================================================
# Total params: 59,046
# Trainable params: 59,046
# Non-trainable params: 0
# _________________________________________________________________


model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=200, epochs = 15, validation_split= 0.2)

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

