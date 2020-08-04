


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Conv2DTranspose, UpSampling2D



def autoencoder(hidden_layer_size) :
    model = Sequential() 
   

    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2,2), 
                     padding= 'valid', input_shape=(28,28,1), activation = 'sin'))
    
    model.add(Conv2D(filters=int(hidden_layer_size/2), kernel_size=(2,2), 
                     padding= 'valid', activation = 'sin'))
    
    model.add(Conv2D(filters=int(hidden_layer_size/4), kernel_size=(2,2), 
                     padding= 'valid', activation = 'sin'))
    
    model.add(Conv2D(filters=int(hidden_layer_size/4), kernel_size=(2,2), 
                     padding= 'valid', activation = 'sin'))
    
    model.add(MaxPool2D(2))
    
    
    model.add(Conv2DTranspose(filters=int(hidden_layer_size/4), kernel_size=(2,2), 
                     padding= 'valid', activation = 'sin'))
    
    model.add(Conv2DTranspose(filters=int(hidden_layer_size/2), kernel_size=(2,2), 
                     padding= 'valid', activation = 'sin'))
    
    # model.add(Conv2DTranspose(filters=hidden_layer_size, kernel_size=(2,2), 
    #                  padding= 'valid', activation = 'sin'))
    
    model.add(UpSampling2D(2))
    
    
    model.add(Conv2D(filters=1, kernel_size=(3,3), 
                     padding= 'same', activation = 'sin'))
    
    model.summary() 
    # Conv2D의 반대는 Conv2DTranspose(가중치 학습 가능), MaxPooling2D의 반대는 UpSampling2D(가중치 학습 x)
    
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data() 

x_train, y_train = train_set 
x_test, y_test = test_set 

#  (60000, 784, 1)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
x_train = x_train/255.
x_test = x_test/255.



model = autoencoder(hidden_layer_size=32)

# model.compile(optimizer = 'adam', loss ='mse', metrics=['acc'])  loss = 0.01
model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics=['acc'])


model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random

fig,((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize = (20, 7))

# 이미지 다섯 개 랜덤 선택 
random_images = random.sample(range(output.shape[0]),5)




# 원본(입력) 이미지를 맨 위에 그린다. 
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0: 
        ax.set_ylabel('INPUT', size =40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    

# 오토인코더가 출력한 이미지를 아래에 그린다 

for i , ax in enumerate([ax6, ax7, ax8, ax9, ax10]) : 
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel('OUTPUT', size =40 )
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show() 

    
