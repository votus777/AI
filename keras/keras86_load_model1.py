
# from keras85_save1.py 

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

model = load_model('./model/model_test01_after.h5')

model.summary()



# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)
# val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  
print('loss :', loss)
print('accuracy : ', acc)



'''

fit 다음에 save 한 파일을 load하면 저장된 가중치 값이 나온다

loss : 0.02036984902289645
accuracy :  0.9940000176429749



하지만 fit하기 전에 save 하고 load를 하면 

RuntimeError: You must compile a model before training/testing. Use `model.compile(optimizer, loss)`. 에러 난다. 

fit과 compile이 되어 있지 않기 때문이다. 86.py 여기서도 fit이 없어서 실행 불가되는 것 



./model/model_test01_after.h5 과 ./model/model_test01_before.h5 을 비교해보면 된다. 


after는 상자와 알맹이 모두 들어있고
before는 상자만 있는 상태 


'''