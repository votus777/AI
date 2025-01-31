
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Concatenate    #    merge1 = concatenate([output1, output2], name = 'merge') 
from keras.layers.merge import                        concatenate    #     Concatenate()([x1, x2])
from keras.callbacks import EarlyStopping, ModelCheckpoint
    
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


################ 데이터 불러오기 #####################

hite = np.load('./data/hite.npy', allow_pickle=True)

samsung = np.load('./data/samsung.npy', allow_pickle=True)

#####################################################

print(hite.shape)  # (508, 5)
print(samsung.shape) # (508, 1)

# =============================
#   TO DO List 

# 1. 데이터 정규화
# 2. 데이터 스플릿  -> 5 : 5
# 3. Input, Output 지정 
# 4. 앙상블 모델 구성
# 5. 가중치,모델 저장
# 6. predict 값 구하기   
# =============================


# # 1. 
minMaxScaler = MinMaxScaler()
hite[ : ,:4] = minMaxScaler.fit_transform(hite[ : , : 4])

# standard_scaler = StandardScaler()   
# hite[ : ,:4] = standard_scaler.fit_transform(hite[ : , : 4])



robustScaler = RobustScaler()
hite[ : , -1 :] = robustScaler.fit_transform(hite[ : , -1 :])

# hite[ : , -1 :] = standard_scaler.fit_transform(hite[ : , -1 :])


# 2. 

def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
    print(type(aaa))
    return np.array(aaa)


dataset_x = split_x(hite,5)
print("=============================")
print(dataset_x)    
# print(dataset_x.shape) # (504, 5, 5)

'''
[0.28037383 0.25917927 0.29567308 0.26652452 0.02363458]  ->  2018-05-04
[0.28271028 0.25701944 0.27884615 0.24520256 0.08661248]  ->  2018-05-08
[0.26401869 0.24190065 0.27644231 0.24733475 0.04427675]  ->  2018-05-09
[0.26869159 0.24406048 0.27403846 0.24307036 0.06914481]  ->  2018-05-10
[0.26168224 0.25701944 0.27884615 0.26012793 0.0763874 ]  ->  2018-05-11
dataset_y = split_x(samsung,5)
print("=============================")
print(dataset_y)
print(dataset_y.shape)  
[53000.]  ->  2018-05-04
[52600.]  ->  2018-05-08
[52600.]  ->  2018-05-09
[51700.]  ->  2018-05-10
[52000.]  ->  2018-05-11
          
여기서 spilt_x 함수를 개조해보자 
전날 5일치 데이터로 다음날 주가를 예측할 수 있도록 
아니면 이틀 뒤까지 
'''
def split_y (samsung, size) :
    aaa = []
    for i in range(len(samsung) - size + 1 ) :
        subset = samsung[ i : (i + size )]
        aaa.append([item for item in subset])  
       
    print(type(aaa))
    return np.array(aaa[3:])

dataset_y = split_y (samsung,5) 
# print(dataset_y)
# print(dataset_y.shape)  # (501, 5, 1)


'''
[53000.]  ->  2018-05-04
[52600.]  ->  2018-05-08
[52600.]  ->  2018-05-09
[51700.]  ->  2018-05-10
[52000.]  ->  2018-05-11
[52600.]   -> 2018-05-08
[52600.]   -> 2018-05-09
[51700.]   -> 2018-05-10
[52000.]   -> 2018-05-11          
[51000.]]  -> 2018-05-14    -> return np.array(aaa[1:])  // 하루 뒤 예측 
[52600.]
[51700.]
[52000.]
[51000.]   -> 2018-05-14
[50200.]   -> 2018-05-15    -> return np.array(aaa[2:])  // 이틀 뒤 예측 
3일 뒤 예측이니 6월 3일을 위해서는 마지막 한 덩이 넣음 되겠다
'''

x = dataset_x[ : -3]   # (504, 5, 5) -> (501, 5, 5)
y = dataset_y

y = y.reshape(501,5)



from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = False  , train_size = 0.8 
)



# 모델 


input1 = Input(shape=(5,5))
dense1 = Dense(18, activation='relu',input_shape =(5,5))(input1)
dense1 = Dense(8, activation='relu')(dense1)
output1 = Dense(8, activation='relu')(dense1)


input2 = Input(shape=(5,5))
dense2 = Dense(18, activation='relu',input_shape =(5,5))(input2)
dense2 = Dense(8, activation='relu')(dense2)
output2 = Dense(8, activation='relu')(dense2)


from keras.layers.merge import concatenate   
merge1 = concatenate([output1, output2], name = 'merge') 

middle1_2 = Dense(5,activation='relu')(merge1)




model = Model(inputs=[input1,input2], outputs = [middle1_2])



# 훈련 

early_stopping = EarlyStopping(monitor='loss', patience= 10, mode ='auto')
modelpath = './model/{epoch:02d} - {val_loss: .4f}.hdf5' 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit([x_train,x_train] ,y_train, validation_split=0.2, verbose=1, batch_size=1, epochs=20, callbacks=[early_stopping])


# model.save('./model/model_0602test.h5') 
# model.save_weights('./model/test_weight1.h5')


#4. 평가, 예측____________________________________________

loss, mse = model.evaluate([x_test,x_test], y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y_pred = model.predict([x_test,x_test])

for i in range(5):
    print('테스트 종가 : ', y_test[i, -1:], '/테스트 예측가 : ', y_pred[i, -1:])

#######################################################################################


t = x[-1]
print(t)

t = t.reshape(1,5,5)

y_0603 = model.predict([t,t])
print('6월 3일 삼성 주가 : ',y_0603[ :,3])

# 튜닝은 아직 하지 않았다 

'''
테스트 종가 :  [56000.] /테스트 예측가 :  [49251.9]
테스트 종가 :  [54900.] /테스트 예측가 :  [49371.46]
테스트 종가 :  [55700.] /테스트 예측가 :  [49322.797]
테스트 종가 :  [56200.] /테스트 예측가 :  [49288.48]
테스트 종가 :  [58400.] /테스트 예측가 :  [49028.195]
6월 3일 삼성 주가 :  [51591.668]
'''

plt.figure(figsize= (10,6))


plt.subplot(2, 1, 1)    # 2행 1열의 첫번쨰 그림을 사용하겠다. 인덱스는 0부터 시작하는데, 이건 아니다.


plt.plot(hist.history['loss'] , marker = '.', c = 'red', label = 'loss')  # plot 추가 =  선 추가 
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')  
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])   # 1st legend : 1st plot,  2nd legend : 2nd plot 
# plt.legend(loc='upper right')
plt.show()

