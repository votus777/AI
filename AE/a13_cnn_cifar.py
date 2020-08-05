
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Conv2DTranspose, UpSampling2D
import numpy as np

def autoencoder(hidden_layer_size) :
    model = Sequential() 
   

    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2,2), 
                     padding= 'valid', input_shape=(32,32,3), activation = 'sin'))
    
    model.add(Conv2D(filters=int(hidden_layer_size/2), kernel_size=(2,2), 
                     padding= 'valid', activation = 'sin'))
    
    model.add(Conv2D(filters=int(hidden_layer_size/4), kernel_size=(3,3), 
                     padding= 'valid', activation = 'sin'))
    
    
    
    model.add(MaxPool2D(2))
    
    
    model.add(Conv2DTranspose(filters=int(hidden_layer_size/4), kernel_size=(2,2), 
                     padding= 'valid', activation = 'sin'))
    
    model.add(Conv2DTranspose(filters=int(hidden_layer_size/2), kernel_size=(2,2), 
                     padding= 'valid', activation = 'sin'))
    
    # model.add(Conv2DTranspose(filters=hidden_layer_size, kernel_size=(2,2), 
    #                  padding= 'valid', activation = 'sin'))
    
    model.add(UpSampling2D(2))
    
    
    model.add(Conv2D(filters=3, kernel_size=(3,3), 
                     padding= 'same', activation = 'sin'))
    
    model.summary() 
    
    return model

from tensorflow.keras.datasets import cifar10

train_set, test_set = cifar10.load_data() 

x_train, y_train = train_set 

x_test, y_test = test_set 


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2],3))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],x_test.shape[2],3))
x_train = x_train/255.
x_test = x_test/255.

#(60000, 32,32,3)

# 노이즈 추가 - 여기서는 배경이 충분하기 때문에 잡음 배제 
# x_train_noised = x_train + np.random.normal( 0, 0.25, size = x_train.shape) # 정규분포로 퍼진 난수
# x_test_noised = x_test + np.random.normal(0 , 0.25, size = x_test.shape )

# x_train_noised = np.clip(x_train_noised, a_min =0, a_max=1)
# x_test_noised = np.clip(x_test_noised, a_min =0, a_max=1)

# np.clip() - array 내의 element들에 대해서 min 값 보다 작은 값들을 min값으로 바꿔주고 
#                                          max 값 보다 큰 값들을 max값으로 바꿔주는 함수.


model = autoencoder(hidden_layer_size=32)

# model.compile(optimizer = 'adam', loss ='mse', metrics=['acc'])  loss = 0.01
model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics=['acc'])


model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random

fig,((ax1, ax2, ax3, ax4, ax5),(ax11,ax12,ax13,ax14,ax15)) = \
        plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개 랜덤 선택 
random_images = random.sample(range(output.shape[0]),5)




# 원본(입력) 이미지를 맨 위에 그린다. 
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(32,32,3), cmap='gray')
    if i == 0: 
        ax.set_ylabel('INPUT', size =40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    

# 오토인코더가 출력한 이미지를 아래에 그린다 

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]) : 
    ax.imshow(output[random_images[i]].reshape(32, 32,3), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel('OUTPUT', size =40 )
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show() 

    
