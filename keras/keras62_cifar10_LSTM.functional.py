
# keras62_cifar10_LSTM.py

# 10가지 컬러 이미지 

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = cifar10.load_data()


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(y_train.shape)  # (60000, 10) -> one hot encoding 

x_train  = x_train.reshape(50000,12,256).astype('float32')/255.0  
x_test  = x_test.reshape(10000,12,256).astype('float32')/255.0  


# ____________모델 구성____________________


model= Sequential()

input = Input(shape=(12, 256), name= 'input_1') 


dense1 = LSTM(256, activation = 'relu', name = 'output_1')(input)
batch_1 = BatchNormalization()(dense1)
dropout1 = Dropout(0.2)(batch_1)


dense1_2 = Dense (64, activation = 'relu',  name = 'output_1_2')(dropout1)
batch_2 = BatchNormalization()(dense1_2)
dropout2 = Dropout(0.2)(batch_2)



dense1_3 = Dense (64, activation = 'relu', name = 'output_1_3')(dropout2)
batch_3 = BatchNormalization()(dense1_3)
dropout3 = Dropout(0.2)(batch_3)


dense1_4 = Dense (32, activation = 'relu', name = 'output_1_4')(dropout3)
batch_4 = BatchNormalization()(dense1_4)
dropout4 = Dropout(0.1)(batch_4)


dense1_5 = Dense (32, activation = 'relu', name = 'output_1_5')(dropout4)
batch_5 = BatchNormalization()(dense1_5)
dropout5 = Dropout(0.1)(batch_5)


output = Dense (10, activation= 'softmax' , name = 'output_1_6')(dropout5)


model = Model (inputs = input, outputs= (output))


model.summary()




# 훈련 


from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['acc'])

model.fit(x_train,y_train, epochs= 20, batch_size= 120, validation_split= 0.25 ,callbacks= [early_stopping])


# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)


