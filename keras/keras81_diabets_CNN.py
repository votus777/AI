

from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn  import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 
import numpy as np

diabets = datasets.load_diabetes()


x = np.array(diabets.data)
y = np.array(diabets.target)

# print(x[0])   # (442, 10)
# print(y)   # (442, )

# print(np.std(y))     # np.mean(y) = 152,   표준편차 : 77.00574586945044  -> 일단  150 이상이면 1이라 둔다. 깊게는 생각 안함 -> 그냥 회귀분석하자 


# 데이터 설명에  data가  이미 표준화 되어있으므로 



print(x)   # (442,10)
print(y)   #(442, )


transformer_Standard = StandardScaler()    
transformer_Standard.fit(x)

x = transformer_Standard.transform(x)


x.sort()

transformer_PCA = PCA(n_components=4)  # PCA 차원 축소 
transformer_PCA.fit(x)

x = x.reshape(442,10,1,1)

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)



# 모델 구성 

model= Sequential()

input0 = Input(shape=(10,1,1))


model_1 = Conv2D(64, (3,3), activation = 'relu', padding= 'same' )(input0)
model_b=BatchNormalization()(model_1)
model_d = Dropout(0.3)(model_b)
model_2 = Conv2D(64, (3,3), activation = 'relu', padding= 'same' )(model_d)
model_b=BatchNormalization()(model_2)
model_d = Dropout(0.3)(model_b)


model_3 = Conv2D(128, (3,3), activation = 'relu', padding= 'same' )(model_d)
model_b=BatchNormalization()(model_3)
model_d = Dropout(0.3)(model_b)


model_4 = Conv2D(128, (5,5), activation = 'relu', padding= 'same' )(model_d)
model_b=BatchNormalization()(model_4)
model_d = Dropout(0.3)(model_b)


model_5 = Conv2D(64, (3,3), activation = 'relu', padding= 'same')(model_d)
model_b=BatchNormalization()(model_5)
model_d = Dropout(0.3)(model_b)



model_7 = Conv2D(32, (3,3), activation = 'relu', padding= 'same')(model_d)
model_b=BatchNormalization()(model_7)
model_d = Dropout(0.3)(model_b)





model_flatten = Flatten()(model_d)                              
model_b=BatchNormalization()(model_flatten)


model_dense = Dense(512, activation='relu')(model_b)
model_b=BatchNormalization()(model_dense)
model_d = Dropout(0.5)(model_b)

model_output = Dense(1, activation= 'relu')(model_d)


model = Model (inputs = input0, outputs= (model_output))

model.summary()

# 훈련 



early_stopping = EarlyStopping( monitor='loss', patience= 50, mode ='auto')

modelpath = './model/{epoch:02d}-{val_loss: .4f}.hdf5' # 02d : 두자리 정수,  .4f : 소수점 아래 4자리 까지 float 실수

checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, mode = 'auto')

tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph= True, write_images= True)

model.compile(loss = 'mse', optimizer='adam', metrics = ['mse'])
hist = model.fit(x_train,y_train, epochs= 10000, batch_size= 5, validation_split= 0.2 , callbacks= [early_stopping])




# 평가 및 예측 


loss, acc = model.evaluate(x_train,y_train, batch_size=1)
val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  
print('loss :', loss)
print('mse : ', mse)
