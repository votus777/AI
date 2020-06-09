
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# 데이터 

train = pd.read_csv('./data/dacon/comp3/train_features.csv', header = 0, index_col = 0)
target = pd.read_csv('./data/dacon/comp3/train_target.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', header = 0, index_col = 0)

# print(train.shape)  # (1050001, 5)
# print(target.shape)  # (2800, 4)
# print(test.shape)   # (262500, 5)
 
# print(type(train))  # <class 'pandas.core.frame.DataFrame'>

# train - target   //   test - summit 

# train x : (2800, 375 ,4 )
# train target : (2800, 4)

# test x : ( 700, 375, 4)
# summit y : (700 ,4 )

# print(train.loc[0])   # [375 rows x 5 columns]

train_x = train.values

train_x = train_x [1 : , 1 :]
print(type(train_x))
print(len(train))
print(train_x.shape)  # (1050000, 4)

train_x = train_x.reshape(2800, 375, 4)

x = train_x[ : , :370]
y = train_x[ : , 370:]


'''
def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
    print(type(aaa))
    return np.array(aaa)


train_x = split_x(train_x,4)
print("=============================")
print(train_x)    


'''

