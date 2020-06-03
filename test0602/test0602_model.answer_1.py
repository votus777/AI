
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Concatenate
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
    
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

################ 데이터 불러오기 #####################


hite = np.load('./data/hite_answer.npy', allow_pickle=True)

samsung = np.load('./data/samsung_answer.npy', allow_pickle=True)


print(samsung.shape)   # (509, 1)
print(hite.shape)      # (509, 5)


######################################################


def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
    # print(type(aaa))
    return np.array(aaa)




samsung = samsung.reshape(samsung.shape[0],)  # (509, )

samsung = split_x(samsung, 6)

print(samsung.shape)  #   (504, 6)

x_sam = samsung [ :, 0:5]
y_sam = samsung [ :, 5]


print(x_sam.shape)  # (504, 5)
print(y_sam.shape)  # (504, )


x_hit = hite[5 : 510, ]
print(x_hit.shape)   # (504,)


# 2. 모델 

input1 = Input(shape=(5,))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,))
x2 = Dense(5)(input2)
x2 = Dense(5)(x2)

merge = Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1,input2], outputs = output)




# 3. 컴파일

model.compile(optimizer='adam', loss = 'mse')
model.fit([x_sam,x_hit], y_sam, epochs= 5)






