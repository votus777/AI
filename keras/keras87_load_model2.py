
# from keras86_save2.py 

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

print(x_train.shape)  #(60000, 28, 28)

print(y_train.shape)  #(60000,)    

print(x_test.shape)   #(10000, 28, 28)
print(y_test.shape)   #(10000,)     


print(x_train[0].shape)   # (28, 28)

# plt.imshow(x_train[0], 'gray')  
# plt.show()  



# _________데이터 전처리 & 정규화 _________________

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10) -> one hot encoding 

x_train  = x_train.reshape(60000,28,28,1).astype('float32')/255.0   # CNN 모델에 input 하기 위해 4차원으로 만들면서 실수형으로 형변환 & 0과 1 사이로 Minmax정규화 
x_test  = x_test.reshape(10000,28,28,1).astype('float32')/255.0      

# ____________모델 구성____________________

from keras.models import load_model
from keras. layers import Dense, BatchNormalization



model = load_model('./model/model_test01_after.h5')


model.add(Dense(10, activation = 'relu', name = 'new_1'))  
model.add(Dense(10, activation = 'softmax', name = 'new_2'))  
model.add(Dense(10, activation = 'softmax', name = 'output'))    




model.summary()









# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)
# val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  
print('loss :', loss)
print('accuracy : ', acc)


'''

loss : 2.313292753124237
accuracy :  0.11349999904632568

0.99 넘게 나오던 acc가 이상하게 나온다 

-> 레이어를 추가함으로써 모델 자체가 틀려졌기 때문에 기존에 있던 가중치 값이 무쓸모가 되는 것 

'''

