
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


hite1 = np.load('./data/hite_answer.npy', allow_pickle=True)

samsung1 = np.load('./data/samsung_answer.npy', allow_pickle=True)

######################################################

standard_scaler = StandardScaler()
hite = standard_scaler.fit_transform(hite1)
samsung = standard_scaler.fit_transform(samsung1)




pca = PCA(n_components=1)
hite = pca.fit_transform(hite)


print(hite.shape) # (508, 1)

def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
    # print(type(aaa))
    return np.array(aaa)



samsung = samsung.reshape(samsung.shape[0],)  # (508, )

samsung = split_x(samsung, 6)
hite = split_x(hite, 5)

samsung = samsung.reshape(504, 6, 1)

x_sam = samsung [ :, 0:5]
y_sam = samsung [ :, 5]
hite = hite[ : -1]

print(y_sam.shape)  


# 2. 모델 

input1 = Input(shape=(5,1))
x1 = LSTM(10, input_shape=(5,1))(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,1))
x2 = LSTM(5, input_shape=(5,1))(input2)
x2 = Dense(5)(x2)

merge = concatenate([x1,x2])

output = Dense(1)(merge)

model = Model(inputs = [input1,input2], outputs = output)




# 3. 컴파일


early_stopping = EarlyStopping(monitor='loss', patience= 10, mode ='auto')
modelpath = './model/{epoch:02d} - {val_loss: .4f}.hdf5' 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)


model.compile(optimizer='adam', loss = 'mse', metrics=['mse'])
model.fit([x_sam,hite], y_sam,  verbose=1, batch_size=1, validation_split=0.2, epochs= 1,  callbacks=[early_stopping])


#4. 평가, 예측____________________________________________
loss, mse = model.evaluate([x_sam,hite], y_sam, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)


t1 = hite1[-1]
t2 = samsung1[-5 :]

t1 = t1.reshape(1,5,1)
t2 = t2.reshape(1,5,1)

y_pred = model.predict([t1,t2])
print(y_pred)

print(t1)
print(t2)

print(y_sam[-1])

# print(x_sam[-1])
# [[0.69745589][0.90071082][0.83973434][1.02266377][1.02266377]]   (5,1)

# print(hite[-1]) 
# [[-0.39250813][-0.54537357][-0.61840762][-0.67756708][-0.51217697]]  (5,1)


# print(samsung1[-5:])       (5,1)
# [[52000][51700][52600][52600][53000]]


# print(hite1[-1])
# [21400 21600 21350 21550 123592]  (1,5)  -> reshape(1,5,1)



