
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn  import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 
import numpy as np
# 회귀모델 

boston = datasets.load_boston()

'''
print(boston.DESCR)



'''
'''
 //결측치  ->  없애거나 채우거나 혹은, 결측치들만 따로 넣어서 predict 하는 방법 

 //이상치  -> 
  

'''
# print(boston.data.shape)  # (506, 13)  -> x

# print(boston.target.shape) #(506,)  -> y

#print(boston.data[:10])

transformer_Standard = StandardScaler()   # 차원 축소 하기 전에 무조건 standard scaler(표준화- 정규분포화) 적용!!  
transformer_Standard.fit(boston.data)


transformer_PCA = PCA(n_components=7)  # PCA 차원 축소 
transformer_PCA.fit(boston.data)


x = transformer_PCA.transform(boston.data)


x_train = x[ :500]
x_test = x[500:]

y_train = boston.target[ : 500]

print(x_train.shape)  # (500,n)
print(x_test.shape)  # (6,n)
print(y_train.shape)  # (500,)

# x_train= x_train.reshape(500,4,1)
# x_test = x_test.reshape(6,4,1)


# LSTM
x_train  = x_train.reshape(500,7).astype('float32')  
x_test  = x_test.reshape(6,7).astype('float32')



# 모델 구성


model= Sequential()

model.add(Dense(32, activation = 'relu', input_shape = (7,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
'''


model.add(LSTM(128, activation = 'relu', input_shape = (4,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
'''

model.add(Dense(64,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))


model.add(Dense(32,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1, activation= 'relu')) 

model.summary()

model.save('./model/sample/boston/model_boston.h5') 



# 훈련 


from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping( monitor='loss', patience= 50, mode ='auto')

model.compile(loss = 'mse', optimizer='adam', metrics = ['mse'])

modelpath = './model/sample/boston{epoch:02d} - {val_loss: .4f}.hdf5' 

checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)

hist = model.fit(x_train,y_train, epochs= 500, batch_size= 5, validation_split= 0.1 ,callbacks= [early_stopping, checkpoint])


model.save_weights('./model/sample/boston/weights_boston.h5')



# 평가 및 예측 


loss, mse = model.evaluate(x_train,y_train, batch_size=1)

y_test = model.predict(x_test)
  
print('loss :', loss)
print('mse : ', mse)


print('y_predict : ', y_test)


# 시각화 

plt.figure(figsize= (10,6))



plt.subplot(2, 1, 1)    

plt.plot(hist.history['loss'] , marker = '.', c = 'red', label = 'loss')  
plt.plot(hist.history['mse'], marker = '.', c = 'blue', label = 'mse')  
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'mse'])   
plt.show()

'''

loss : 35.62095818808851
mse :  35.620933532714844
y_predict :  


[[20.01724 ]
 [21.957827]
 [22.283232]
 [26.878468]
 [25.589577]
 [22.938572]]


 '''