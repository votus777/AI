
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

import matplotlib.pyplot as plt 
import random 

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


model_8 = autoencoder(hidden_layer_size=8)
model_16= autoencoder(hidden_layer_size=16)
model_32= autoencoder(hidden_layer_size=32)
model_64= autoencoder(hidden_layer_size=64)
model_128= autoencoder(hidden_layer_size=128)
model_254= autoencoder(hidden_layer_size=254)



model_8.compile(optimizer = 'adam', loss ='mse', metrics=['acc'])  
model_16.compile(optimizer = 'adam', loss ='mse', metrics=['acc'])  
model_32.compile(optimizer = 'adam', loss ='mse', metrics=['acc'])  
model_64.compile(optimizer = 'adam', loss ='mse', metrics=['acc'])  
model_128.compile(optimizer = 'adam', loss ='mse', metrics=['acc'])  
model_254.compile(optimizer = 'adam', loss ='mse', metrics=['acc'])  



model_8.fit(x_train, x_train, epochs=10)
model_16.fit(x_train, x_train, epochs=10)
model_32.fit(x_train, x_train, epochs=10)
model_64.fit(x_train, x_train, epochs=10)
model_128.fit(x_train, x_train, epochs=10)
model_254.fit(x_train, x_train, epochs=10)

output_8 = model_8.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)
output_64 = model_64.predict(x_test)
output_128 = model_128.predict(x_test)
output_254 = model_254.predict(x_test)


# 그림을 그리자 
fig, axes = plt.subplots( 7, 5, figsize = (15, 15))

random_imgs = random.sample(range(output_8.shape[0]),5) 
outputs = [x_test, output_8, output_16, output_32, output_64, output_128, output_254]

for row_num, row in enumerate(axes) :
    for col_num, ax in enumerate(row) :
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28),cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show() 

