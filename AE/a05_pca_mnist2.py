
# keras56_mnist_DNN.py
# input_dim 154 


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()


# plt.imshow(x_train[0], 'gray')  
# plt.show()  


# _________데이터 전처리 & 정규화 _________________

x_train  = x_train.reshape(60000,784).astype('float32')/255.0    
x_test  = x_test.reshape(10000,784).astype('float32')/255.0  

print(x_train.shape)  #(60000, 784)
print(x_test.shape)   #(10000, 784)      # Dense 모델에 맞게 reshape 


from sklearn.decomposition import PCA

pca = PCA(n_components=154)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)



# ____________모델 구성____________________

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten

model= Sequential()

model.add(Dense(256, activation = 'sin', input_shape = (154,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256,activation = 'sin'))
model.add(BatchNormalization())
model.add(Dropout(0.3))


model.add(Dense(512,activation = 'sin'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Dense(256,activation = 'sin'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1, activation= 'sigmoid')) 

model.summary()


# 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train,y_train, epochs= 15, batch_size= 254, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=100)

  
print('loss :', loss)
print('accuracy : ', acc)