

from keras.preprocessing.text import Tokenizer 

import numpy as np

docs = ["인생 너무 재밌다", "삼겹살 최고", "곱창도 먹고 싶다", "내가 구운 삼겹살", "스위스에 사는 스미스씨의 수비드 삼겹살", "엿먹어라 라스트오브어스2"]

# 긍정 = 1, 부정 = 2
labels = np.array([0,1,1,0,1,0])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)

word_size = len(token.word_index) 

x = token.texts_to_sequences(docs)

from keras.preprocessing import sequence
pad_x = sequence.pad_sequences(x, padding='post', value=0)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D

model = Sequential()

# embedding 뺴고 LSTM 만으로 구성 

# model.add( Embedding( word_size, 10, input_length = 5))  

print(pad_x.shape)  # (6, 5)

pad_x = pad_x.reshape(6,5,1)
                 

# model.add(LSTM(3, input_shape = (5,1))) 
model.add(Conv1D(10,1, input_shape = (5,1)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))  



# model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print('acc : ' , acc)

