#keras46_hist.py



import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 

a = np.array (range(1,101))
size = 5   # time_steps = 4


def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])   
   
    return np.array(aaa)


dataset = split_x(a,size) #(6,5)
print(type(dataset)) 

x =dataset [ :, 0:4]     
y =dataset [ :, 4]


x = x.reshape(96,4,1)
y = y.reshape(96,1)



# 2. 모델 구성 

from keras.models import load_model 
model = load_model('./model/save_keras44.h5')



model.add(Dense(20,name = 'new_1'))  
model.add(Dense(20,name = 'new_2'))  


model.add(Dense(1,name = 'out'))  

# model.summary()


# 3. 훈련
from keras.callbacks import EarlyStopping  
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse', metrics = ['acc'])
hist = model.fit(x,y, epochs= 10000, batch_size = 1, callbacks= [ealry_stopping], validation_split = 0.2 )

print(hist)
print(hist.history.keys())


import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])   # plot 추가 =  선 추가 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['val_loss'])

plt.title('loss & acc')
plt.xlabel('epoch')
plt.ylabel('loss,acc')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()





#. 4. 평가 및 예측 

loss,mse = model.evaluate(x, y, batch_size = 1) 

x_predict  = np.array(x) 
y_predict = model.predict(x)

print('loss :', loss)
print('mse :', mse)
