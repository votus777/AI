import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 

a = np.array(range(1,11)) 
size = 4


def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)


dataset = split_x(a,size)
print("=============================")
print(dataset)
print(dataset.shape)
'''

size = 5    // 여기서 size는 곧 LSTM에서 tiem_steps이다.  

<class 'list'>
=============================
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

----------------------------------


size = 4

<class 'list'>  
=============================
[[ 1  2  3  4]
 [ 2  3  4  5]
 [ 3  4  5  6]
 [ 4  5  6  7]
 [ 5  6  7  8]
 [ 6  7  8  9]
 [ 7  8  9 10]]



 ''' 