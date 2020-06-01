

from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.datasets  import load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt 
import numpy as np

diabets = load_diabetes()


x = np.array(diabets.data)
y = np.array(diabets.target)


print(x.shape)  # (442, 10)
print(y.shape)  # (442, )



#########################################################

transformer_Standard = StandardScaler()    
transformer_Standard.fit(x)

x = transformer_Standard.transform(x)
# y = transformer_Standard.transform(y)   -> 이거만 하면 에러 뜬다. ValueError: non-broadcastable output operand with shape (442,1) doesn't match the broadcast shape (442,10)

minmax_scaler = MinMaxScaler()

y = y.reshape(442,1)

x = minmax_scaler.fit_transform(x)
y = minmax_scaler.fit_transform(y)     #  -> 이건 가능하다. 

######################################################

x = x.reshape(442,10,1,1)



from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)



# 모델 구성 

model= Sequential()

input0 = Input(shape=(10,1,1))


model_1 = Conv2D(32, (3,3), activation = 'relu', padding= 'same' )(input0)
model_d = Dropout(0.3)(model_1)
model_2 = Conv2D(32, (3,3), activation = 'relu', padding= 'same' )(model_d)
model_d = Dropout(0.3)(model_2)


model_3 = Conv2D(16, (3,3), activation = 'relu', padding= 'same' )(model_d)
model_d = Dropout(0.3)(model_3)


model_4 = Conv2D(32, (5,5), activation = 'relu', padding= 'same' )(model_d)
model_d = Dropout(0.3)(model_4)


model_5 = Conv2D(32, (3,3), activation = 'relu', padding= 'same')(model_d)
model_d = Dropout(0.3)(model_5)



model_7 = Conv2D(16, (3,3), activation = 'relu', padding= 'same')(model_d)
model_d = Dropout(0.3)(model_7)





model_flatten = Flatten()(model_d)                              
model_b=BatchNormalization()(model_flatten)


model_dense = Dense(128, activation='relu')(model_b)
model_d = Dropout(0.5)(model_dense)

model_output = Dense(1, activation= 'relu')(model_d)


model = Model (inputs = input0, outputs= (model_output))

model.summary()

# 훈련 



early_stopping = EarlyStopping( monitor='loss', patience= 10, mode ='auto')

modelpath = './model/{epoch:02d}-{val_loss: .4f}.hdf5' # 02d : 두자리 정수,  .4f : 소수점 아래 4자리 까지 float 실수

checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, mode = 'auto')

tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph= True, write_images= True)

model.compile(loss = 'mse', optimizer='adam', metrics = ['mse'])
hist = model.fit(x_train,y_train, epochs= 10000, batch_size= 5, validation_split= 0.2 , callbacks= [early_stopping])




# 평가 및 예측 


loss, mse = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('mse : ', mse)

'''

loss : 0.058078822099151294
mse :  0.058078814297914505

'''