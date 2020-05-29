
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn  import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 
import numpy as np



# 데이터 

iris = datasets.load_iris()

x = iris.data
y = iris.target


print(x.shape)  # (150,4)
print(y.shape)  # (150,)

# print(x[:10])
# print(y)   # 0~ 50 : 0 // 51 ~ 100 : 1 // 101 ~ 150 : 2  

# 데이터가 순차적으로 섞여있다. 


transformer_Standard = StandardScaler()    # 정규화
transformer_Standard.fit(x)


x = x.reshape(150,4,1,1)                      # LSTM reshape

y= np_utils.to_categorical(y)               # One-Hot-encoding



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)

print(y_train)


# 모델 

model = Sequential()
model.add(Conv2D(10,(1,1), activation='relu', input_shape = (4,1,1)))

model.add(Flatten())

model.add(Dense(50, activation= 'softmax')) 
model.add(Dense(3, activation= 'softmax')) 



# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

hist = model.fit(x,y, epochs= 10000, batch_size= 1, validation_split= 0.25,  callbacks= [early_stopping])



# 평가 및 예측 


loss, acc = model.evaluate(x,y, batch_size=1)


print('loss :', loss)
print('accuracy : ', acc)


# 시각화 

plt.figure(figsize= (10,6))



plt.subplot(2, 1, 1)    

plt.plot(hist.history['loss'] , marker = '.', c = 'red', label = 'loss')  
plt.plot(hist.history['acc'], marker = '.', c = 'blue', label = 'acc')  
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['loss', 'acc'])   
plt.show()
