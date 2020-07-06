


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
from keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D, MaxPooling1D

model = Sequential()

model.add(Embedding(16,10, input_shape = (5,)))                 
model.add(Conv1D(10,1))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))  

# 즉,  Embedding 과 LSTM 동시에 쓰면 input_length 를 명시해 주지 않아도 되고 Conv1D 에서는 Flatten 써야 하기 때문에 shape 명시해줘야 하나봄
# LSTM은 3차원으로 받아서 2차원으로 내보내지만 ConV1D는 3차원으로 받아서 3차원으로 내뱉기 때문 




# model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print('acc : ' , acc)

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 5, 10)             160
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 5, 10)             110
_________________________________________________________________
flatten_1 (Flatten)          (None, 50)                0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51
=================================================================
Total params: 321
Trainable params: 321
Non-trainable params: 0
_________________________________________________________________

'''