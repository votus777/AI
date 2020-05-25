#keras45_load.py



import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 

a = np.array (range(1,11))
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


x = x.reshape(6,4,1)
y = y.reshape(6,1)



# 2. 모델 구성 

from keras.models import load_model 
model = load_model('.\model\save_keras44.h5')



model.add(Dense(10,name = 'new_1'))  
model.add(Dense(10,name = 'new_2'))  


model.add(Dense(1,name = 'out'))  

'''

# 현재 원 모델은 output shape 가 10이다. 이걸 고치기 위해 여기서 레이어를 추가했다.
# 그런데 error 메세지를 보면 같은 레이어 두 개가 있다고 뜬다. 
# 그래서 그냥 레이어 이름을 바꿔주었다. 

# model.summary()를 해보면 이름이 같아서 충돌 나는 것을 확인할 수 있었따. 

'''
model.summary()


# 3. 훈련
from keras.callbacks import EarlyStopping  
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, batch_size = 1, callbacks= [ealry_stopping])

#. 4. 평가 및 예측 

x_predict  = np.array([11,12,13,14]) 
x_predict = np.reshape(x_predict,(1,4,1)) 

y_predict = model.predict(x_predict)


print(y_predict)      

