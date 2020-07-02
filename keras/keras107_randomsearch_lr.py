# 100번 카피 복붙 lr 넣고 튜닝 



import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model 
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, MaxPool2D, LSTM


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


def bulid_model(drop=0.5, optimizer = 'adam', learning_rate = 0.08, epoch = 50) :  # 여기에도 learning_rate,epoch 변수 추가 해준다 

    inputs = Input(shape=(784, ), name= 'inputs')  
    x = Dense(64, activation=tf.math.sin, name= 'hidden1', )(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=tf.math.sin, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation=tf.math.sin, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=tf.math.sin, name = 'hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(9, activation='softmax', name= 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss = 'categorical_crossentropy')
    return model

def create_hyperparameters() : 
    batches = [50, 100, 200, 300, 500]
    optimizers = [ 'rmsprop', 'adam', 'adadelta', 'nadam']
    learning_rate = [ 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]  # keras107 -> learning rate parameter 추가 
    dropout = [ 0.1, 0.2, 0.3, 0.4, 0.5]    # start ~ end 사이의 값을 개수만큼 생성하여 배열로 반환합니다.
    epoch = [ 200,400,800, 1000]
    # epoch, node 개수, activation, etc..
    return{"batch_size" :  batches, "optimizer": optimizers, "learning_rate" : learning_rate, "drop" : dropout, 'epoch' : epoch }  # girdsearch 가 dictionary 형태로 값을 받기 때문에 return도 dict형태로 맞춰준다 


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

 - 1s - loss: 2.6658 - acc: 0.2784
{'optimizer': 'rmsprop', 'learning_rate': 0.005, 'epoch': 40, 'drop': 0.5, 'batch_size': 300}
val_scores :  [0.40281942 0.55625564 0.4938494 ]
acc :  0.20409999787807465


activation을 relu 에서 sin으로 바꾸었을 뿐인데..

{'optimizer': 'adam', 'learning_rate': 0.01, 'epoch': 400, 'drop': 0.2, 'batch_size': 50}
val_scores :  [0.74685061 0.77287728 0.78127813]
acc :  0.8402000069618225
'''