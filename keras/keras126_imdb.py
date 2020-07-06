

from keras.datasets import imdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. 데이터 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)




# print(x_train[0])
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 
#  36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 
#  336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14,
# 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 
#  515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 2, 16, 480, 66, 
#  3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 
# 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 
# 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 
# 297, 98, 32, 2071, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 
# 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 
# 113, 103, 32, 15, 16, 2, 19, 178, 32]

# print(y_train[0])  # 1




# get_word_index() 에 인덱스를 집어넣으면 전처리 전에 어떤 단어였는지 확인할 수 있다
# word_to_index = imdb.get_word_index()
# index_to_word={}
# for key, value in word_to_index.items():
#     index_to_word[value] = key

# print('빈도수 상위 1번 단어 : ', index_to_word[1])  # the

# 혹은 reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))





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

model = Sequential()

model.add(Embedding(1000, 128, input_length = 500))   
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))



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

