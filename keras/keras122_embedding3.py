

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
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()

# model.add( Embedding( word_size, 10, input_length = 5))  
model.add(Embedding(16,10)) # 이대로 실행하면 shape 틀렸다는 error가 뜬다...그런데 우리는 아예 shape를 주지 않았는데..?
                 
# model.add(Flatten())
model.add(LSTM(3))  # 심지어 flattten 빼고 LSTM 넣으니 훈련이 된다 
model.add(Dense(1, activation='sigmoid'))  

# 즉,  Embedding 과 LSTM를  동시에 쓰면 input_length 를 명시해 주지 않아도 된다. 
# 이건 embedding 의 특성, vetorziation (벡터화) 때문이다 

# Hands On 525p
# King - Mam + Woman   -> Queen 
# Madrid - Spain + France    -> Paris 
# 다만 이는 편향성을 지닐 수도 있다 ex) Doctor -> Man , Nurse -> Woman 



# model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print('acc : ' , acc)

'''
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 10)          160  
_________________________________________________________________
lstm_1 (LSTM)                (None, 3)                 168 =  4 * (10 + 3 + 1) * 3 
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 4
=================================================================
Total params: 332
Trainable params: 332
Non-trainable params: 0
_________________________________________________________________
'''




