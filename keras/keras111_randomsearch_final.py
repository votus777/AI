

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model 
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, MaxPool2D, LSTM

sin = tf.math.sin
# RandomizedSearchCV 

# 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)

print(y_train.shape)   # (60000,)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1 ).astype('float')/255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1 ).astype('float')/255

x_train = x_train.reshape(x_train.shape[0], 784 )/255
x_test = x_test.reshape(x_test.shape[0], 784 )/255



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

y_train = y_train[ :, 1:]
y_test = y_test[ :, 1:]

print(y_train.shape)  # (60000, 9)
print(y_test.shape)  # (10000, 9)


# 모델 


def bulid_model(drop, optimizer, learning_rate, epochs, activation) :  # 여기에도 learning_rate,epoch 변수 추가 해준다 

    inputs = Input(shape=(784, ), name= 'inputs')  
    x = Dense(64, activation=activation, name= 'hidden1', )(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name = 'hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(9, activation='softmax', name= 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss = 'categorical_crossentropy')
    return model

def create_hyperparameters() : 
    batches = np.linspace(100,1000,10).tolist()
    optimizers = [ 'rmsprop', 'adam', 'adadelta', 'nadam']
    learning_rate = [ 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]  # keras107 -> learning rate parameter 추가 
    dropout = np.linspace(0.1,0.8,8).tolist()    # start ~ end 사이의 값을 개수만큼 생성하여 배열로 반환합니다.
    epochs = np.linspace(100,1000,10).tolist()
    activation = [ 'relu', 'elu', 'tanh', 'selu']
    # epoch, node 개수, activation, etc..
    return{"batch_size" :  batches, "optimizer": optimizers, "learning_rate" : learning_rate, "drop" : dropout, 'epochs' : epochs, 'activation' : activation }  # girdsearch 가 dictionary 형태로 값을 받기 때문에 return도 dict형태로 맞춰준다 

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  

model = KerasClassifier(build_fn=bulid_model, verbose = 2) 

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
search = RandomizedSearchCV(model, hyperparameters, cv=3  )   
search.fit(x_train,y_train)


from sklearn.model_selection import cross_val_score
val_scores = cross_val_score(model,x_test,y_test, cv=3)

acc = search.score(x_test,y_test)

print(search.best_params_)
print("val_scores : " ,val_scores)
print( "acc : ", acc)



'''

activation function 추가 
{'optimizer': 'adam', 'learning_rate': 0.01, 'epochs': 20, 'drop': 0.1, 'batch_size': 300, 'activation': 'tanh'}
val_scores :  [nan nan nan]
acc :  0.875
3.68
'''