
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Concatenate, Conv2D, Flatten
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

print(type(samsung)) # <class 'numpy.ndarray'>
print(type(hite))


print(samsung[10])
print(hite)

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


# print(x_sam.shape)  # (504, 5)
# print(y_sam.shape)  # (504, )

print(x_sam)

x_hit = hite[5 : 510, ]
print(x_hit.shape)   # (504,)

x_sam = x_sam.reshape(504,5,1,1)
x_hit = x_hit.reshape(504,5,1,1)


# 2. 모델 

input1 = Input(shape=(5,1,1))
x1 = Conv2D(10,(1,1) ,input_shape=(5,1,1))(input1)
x1 = Flatten()(x1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,1,1))
x2 = Conv2D(5,(1,1), input_shape=(5,1,1))(input2)
x2 = Flatten()(x2)
x2 = Dense(5)(x2)

merge = concatenate([x1,x2])

output = Dense(1)(merge)

model = Model(inputs = [input1,input2], outputs = output)




# 3. 컴파일


early_stopping = EarlyStopping(monitor='loss', patience= 10, mode ='auto')
modelpath = './model/{epoch:02d} - {val_loss: .4f}.hdf5' 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)


model.compile(optimizer='adam', loss = 'mse', metrics=['mse'])
model.fit([x_sam,x_hit], y_sam,  verbose=1, batch_size=1, validation_split=0.2, epochs= 1,  callbacks=[early_stopping])


#4. 평가, 예측____________________________________________
loss, mse = model.evaluate([x_sam,x_hit], y_sam, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

