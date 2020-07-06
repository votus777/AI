
from keras.preprocessing.text import Tokenizer 

text = "나는 아침밥을 먹고 싶다"

token = Tokenizer()
token.fit_on_texts([text])

# print('token word index : ', token.word_index)
# {'나는': 1, '아침밥을': 2, '먹고': 3, '싶다': 4}
# 단어 단위로 자르고 인덱싱을 걸어준다 

x = token.texts_to_sequences([text])  
# print('x :' , x)
# [[1, 2, 3, 4]]   
# -> 그러나 4번 단어 '싶다'가 ' 1번 단어 '나는' 보다 4배 더 중요한건 아니다, 인덱싱에 가치가 아니라 범주형으로 다루어야 한다. ex) one-hot-encoding

from keras.utils import to_categorical

word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes= word_size)

# print(x)
# [[[0. 1. 0. 0. 0.]
#   [0. 0. 1. 0. 0.]
#   [0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 1.]]]

# 그러나 이런 식으로 접근하면 10만 단어가 될 경우 칼럼 수가 기하급수적으로 커지게 된다 

# 그래서 이것을 압축하자 -> 'embedding' 



