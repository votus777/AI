import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model 
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, MaxPool2D



# 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)

print(y_train.shape)   # (60000,)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1 ).astype('float')/255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1 ).astype('float')/255

x_train = x_train.reshape(x_train.shape[0], 28*28 )/255
x_test = x_test.reshape(x_test.shape[0], 28*28 )/255



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

y_train = y_train[ :, 1:]
y_test = y_test[ :, 1:]

print(y_train.shape)  # (60000, 9)
print(y_test.shape)  # (10000, 9)


# 모델 


def bulid_model(drop=0.5, optimizer = 'adam') :

    inputs = Input(shape=(28*28, ), name= 'inputs')  # (786,)
    x = Dense(512, activation='relu', name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(9, activation='softmax', name= 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss = 'categorical_crossentropy')
    return model

def create_hyperparameters() : 
    batches = [100, 200, 300, 400, 500]
    optimizers = [ 'rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)    # start ~ end 사이의 값을 개수만큼 생성하여 배열로 반환합니다.
    return{"batch_size" :  batches, "optimizer": optimizers, "drop" : dropout }  # girdsearch 가 dictionary 형태로 값을 받기 때문에 return도 dict형태로 맞춰준다 


from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  
# keras.wrappers.scikit_learn.py의 래퍼를 통해 Sequential 케라스 모델을 (단일 인풋에 한정하여) Scikit-Learn 작업의 일부로 사용할 수 있습니다.

model = KerasClassifier(build_fn=bulid_model, verbose = 1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
search = GridSearchCV(model, hyperparameters, cv=3)
search.fit(x_train,y_train)

print(search.best_params_)
