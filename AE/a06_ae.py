
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

def autoencoder(hidden_layer_size) :
    model = Sequential() 
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation = 'sin'))
    model.add(Dense(units=784, activation ='sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data() 

x_train, y_train = train_set 

x_test, y_test = test_set 


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]* x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
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
    ax.imshow(x_test[random_images[i]], cmap='gray')
    if i == 0: 
        ax.set_ylabel('INPUT', size =40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    

# 오토인코더가 출력한 이미지를 아래에 그린다 

for i in ax in enumerate([ax6, ax7, ax8, ax9, ax10]) : 
    ax.imshow(0(output[random_images[i]].reshape(28, 28), cmap = 'gray'))
    if i == 0 :
        ax.set_y_label('OUTPUT', size =40 )
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show() 

    
