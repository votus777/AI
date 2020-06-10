# iris를 케라스 파이프라인 구성
# 당연히 RandomizedSearchCV 구성 


import numpy as np


from keras.utils import np_utils
from keras.models import Model 
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, MaxPool2D


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# RandomizedSearchCV 

# 데이터 
iris = load_iris()

x = iris.data    
y = iris.target  

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 58)

  
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10)
print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

y_train = y_train[ :, 1:]
y_test = y_test[ :, 1:]


# 모델 


def bulid_model(drop=0.5, optimizer = 'adam') :

    inputs = Input(shape=(150), name= 'inputs')  # (786,)
    x = Dense(512, activation='relu', name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(4, activation='softmax', name= 'outputs')(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss = 'categorical_crossentropy')
    return model

def create_hyperparameters() : 
    batches = [ 30, 40, 50]
    optimizers = ['rmsprop']
    dropout = [0.1, 0.3,0.5 ]
   
    return{"models__batch_size" :  batches, "models__optimizer": optimizers, "models__drop" : dropout }
    


from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  

model = KerasClassifier(build_fn=bulid_model, verbose=1)

pipe = Pipeline([("scaler", MinMaxScaler()), ("models", model)])

hyperparameters =create_hyperparameters()


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
search = RandomizedSearchCV(model, hyperparameters, cv=3  )  
search.fit(x_train,y_train)



from sklearn.model_selection import cross_val_score
val_scores = cross_val_score(model,x_test,y_test, cv=3)

acc = search.score(x_test,y_test)

print(search.best_estimator_)
print("val_scores : " ,val_scores)
print( "acc : ", acc)
