
from keras.preprocessing.text import Tokenizer 

import numpy as np

docs = ["인생 너무 재밌다", "삼겹살 최고", "곱창도 먹고 싶다", "내가 구운 삼겹살", "스위스에 사는 스미스씨의 수비드 삼겹살", "엿먹어라 라스트오브어스2"]

# 긍정 = 1, 부정 = 2
labels = np.array([0,1,1,0,1,0])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index) 

# {'삼겹살': 1, '인생': 2, '너무': 3, '재밌다': 4, '최고': 5, 
#  '곱창도': 6, '먹고': 7, '싶다': 8, '내가': 9, '구운': 10, 
#  '스위스에': 11, '사는': 12, '스미스씨의': 13, '수비드': 14, 
#  '엿먹어라': 15, '라스트오브어스2': 16}

# Tokenizer -> 중복된 애들 합쳐짐 -> 수치화 및 압축

word_size = len(token.word_index) 
print('word size : ', word_size ) 
# word size :  16



x = token.texts_to_sequences(docs)
print(x)

# [[2, 3, 4], [1, 5], [6, 7, 8], [9, 10, 1], [11, 12, 13, 14, 1], [15, 16]]
# 여러번 나오는 단어가 앞 번호로 나옴
# 그런데 이대로 input 하기에는 shape가 맞지 않는다 

# pad_sequence 로 padding을 해주는데
# 시계열에서는 가장 최근의 데이터가 더 연관성이 높으니까 의미있는 숫자가 더 큰 가중치를 갖게 하기 위해
# 통상적으로 0을 뒷자리가 아닌 앞자리에 채운다 

from keras.preprocessing import sequence

pad_x = sequence.pad_sequences(x, padding='pre', value=0)
# print(padding)
# [[ 0  0  2  3  4]
#  [ 0  0  0  1  5]
#  [ 0  0  6  7  8]
#  [ 0  0  9 10  1]
#  [11 12 13 14  1]
#  [ 0  0  0 15 16]]

pad_x = sequence.pad_sequences(x, padding='post', value=0)
# print(padding)
# [[ 2  3  4  0  0]
#  [ 1  5  0  0  0]
#  [ 6  7  8  0  0]
#  [ 9 10  1  0  0]
#  [11 12 13 14  1]
#  [15 16  0  0  0]]

# padding = sequence.pad_sequences(x, padding='post', value=1)
# print(padding)
# [[ 2  3  4  1  1]
#  [ 1  5  1  1  1]
#  [ 6  7  8  1  1]
#  [ 9 10  1  1  1]
#  [11 12 13 14  1]
#  [15 16  1  1  1]]

# print(pad_x.shape)  # (6,5)

# vetor화 
# 즉 유사도가 있는 놈끼리 가까이 위치시키게 된다 

# 좌표 자체를 가지고 오는건 의미가 없기 때문에
# 모델 안에서 자체적으로 처리해서 계산하게 된다. 

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
 
model.add( Embedding( word_size, 10, input_length = 5))  # shape 맟춰주는 단계   # ( None, 5, 10) -> Con1D, LSTM etc..
                    # (전제 단어의 숫자-> 벡터화의 갯수 , 노드 갯수 , input_dim)
                    
# model.add( Embedding( 2500, 10, input_length = 5))  # word size를 임의로 바꿔도 잘 돌아간다? 
# 벡터화 시키는 크기를 임의로 줄 수 있다. 
# 단지 params가 달라질 뿐   

#model.add(Embedding(25,10))


model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))  # 이 단어들이 긍정(1)인지 부정(0)인지 판단

     
model.summary()

'''
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 5, 10)             160   = word_size * nodes   , output으로 3차원을 출력해준다 
_________________________________________________________________
flatten_1 (Flatten)          (None, 50)                0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51
=================================================================
Total params: 221
Trainable params: 221
Non-trainable params: 0
_________________________________________________________________

'''

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print('acc : ' , acc)


