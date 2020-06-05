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

# 데이터 불러오기______________________________________________________ 


hite = np.load('./data/hite_answer.npy', allow_pickle=True)

samsung = np.load('./data/samsung_answer.npy', allow_pickle=True)


# Train // Test // Predict 구분______________________________________________

hite = hite[ :-5]
samsung = samsung[ :-5]

hite_predict = hite[ -5 : ]
samsung_predict = samsung[-5 : ]


from sklearn.model_selection import train_test_split
hite_train,hite_test, samsung_train, samgsung_test = train_test_split(
   
    hite, samsung, shuffle = True  , train_size = 0.8  
)

 
# print(hite_train.shape)      # (403, 5)
# print(samsung_train.shape)   # (403, 1)

# print(hite_test.shape)       # (101, 5)
# print(samgsung_test.shape)   # (101, 1)

# print(hite_predict.shape)    # (5 ,5)
# print(samsung_predict.shape) # (5, 1)


# 정규화_________________________________________________________

standard_scaler = StandardScaler()
hite_train = standard_scaler.fit_transform(hite_train)
samsung_train = standard_scaler.fit_transform(samsung_train)

hite_test = standard_scaler.transform(hite_test)
samgsung_test = standard_scaler.transform(samgsung_test)


# 차원 축소______________________________________________________

pca = PCA(n_components=1)
hite_train = pca.fit_transform(hite_train)
hite_test = pca.fit_transform(hite_test)
hite_predict = pca.fit_transform(hite_predict)

# 데이터 스플릿___________________________________________________

def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
    # print(type(aaa))
    return np.array(aaa)



samsung_train = split_x(samsung_train, 6)
hite_train = split_x(hite_train, 5)

samgsung_test = split_x(samgsung_test, 6)
hite_test = split_x(hite_test, 5)

# print(samsung_train.shape) # (398, 6, 1)
# print(hite_train.shape)    # (399, 5, 1)
 
# print(samgsung_test.shape)   # (96, 6, 1)
# print(hite_test.shape)       # (97, 5, 1)


x_sam = samsung_train [ :, 0:5]  
y_sam = samsung_train [ :, -1:]
hite_train = hite_train[ : -1]

x_test_sam = samgsung_test[ : , 0:5]
y_test_sam = samgsung_test[ : , -1:]
hite_test = hite_test [ : -1] 

# print(x_sam.shape)         # (398, 5, 1)
# print(hite_train.shape)    # (398, 5, 1)
# print(y_sam.shape)         # (398, 1, 1)


# print(x_test_sam.shape)    # (96, 5, 1)
# print(hite_test.shape)     # (96, 5, 1)
# print(y_test_sam.shape)    # (96, 1, 1)

y_sam = y_sam.reshape(398,1)
y_test_sam = y_test_sam.reshape(96,1)


# print(hite_predict.shape)    #  ( 5, 1)
# print(samsung_predict.shape) #  ( 5, 1)

hite_predict = hite_predict.reshape(1,5,1)
samsung_predict = samsung_predict.reshape(1,5,1)


# 모델________________________________________________ 

input1 = Input(shape=(5,1))
x1 = LSTM(12, activation='relu',input_shape=(5,1))(input1)
x1 = Dropout(0.4)(x1)

x1 = Dense(8, activation='relu')(x1)
x1 = Dropout(0.4)(x1)

x1 = Dense(8, activation='relu')(x1)
x1 = Dropout(0.4)(x1)


input2 = Input(shape=(5,1))
x2 = LSTM(12, activation='relu',input_shape=(5,1))(input2)
x2 = Dropout(0.4)(x2)

x2 = Dense(4, activation='relu')(x2)
x2 = Dropout(0.4)(x2)

x2 = Dense(8, activation='relu')(x2)
x2 = Dropout(0.4)(x2)


merge = concatenate([x1,x2])

output = Dense(1)(merge)

model = Model(inputs = [input1,input2], outputs = output)




# 컴파일 및 훈련__________________________________________________________________


early_stopping = EarlyStopping(monitor='loss', patience= 10, mode ='auto')
modelpath = './model/{epoch:02d} - {val_loss: .4f}.hdf5' 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)

# model.load_weights('./model/03 -  0.8877.hdf5') 

model.compile(optimizer='adam', loss = 'mse', metrics=['mse'])
hist = model.fit([x_sam,hite_train], y_sam,  verbose=1, batch_size=1, validation_split=0.25, epochs= 100 , callbacks=[early_stopping])


#4. 평가, 예측_____________________________________________________________________
loss, mse = model.evaluate([x_test_sam,hite_test], y_test_sam, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)


y_predict = model.predict([hite_predict,samsung_predict])

print("y_predict : ", y_predict)

# 히스토리______________________________________ 


plt.figure(figsize= (10,6))


plt.subplot(2, 1, 1)    
plt.plot(hist.history['loss'] , marker = '.', c = 'red', label = 'loss')  
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')  
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])   
plt.show()