
# keras60_cifar10_cnn.py

# 10가지 컬러 이미지 

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, MaxPool2D
from keras.layers.merge import concatenate 
import matplotlib.pyplot as plt



# 데이터 전처리 및 정규화

(x_train, y_train),(x_test,y_test) = cifar10.load_data()


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (60000, 10) -> one hot encoding 

x_train  = x_train.reshape(50000,32,32,3).astype('float32')/255.0  
x_test  = x_test.reshape(10000,32,32,3).astype('float32')/255.0  


# ____________모델 구성____________________


model= Sequential()

input0 = Input(shape=(32,32,3))


model_1 = Conv2D(64, (3,3), activation = 'relu', padding= 'same' )(input0)
model_b=BatchNormalization()(model_1)
model_d = Dropout(0.3)(model_b)
model_2 = Conv2D(64, (3,3), activation = 'relu', padding= 'same' )(model_d)
model_b=BatchNormalization()(model_2)
model_d = Dropout(0.3)(model_b)
model_m = MaxPool2D(2,2)(model_d)

model_3 = Conv2D(128, (3,3), activation = 'relu', padding= 'same' )(model_m)
model_b=BatchNormalization()(model_3)
model_d = Dropout(0.3)(model_b)
model_m = MaxPool2D(2,2)(model_d)

model_4 = Conv2D(128, (3,3), activation = 'relu', padding= 'same' )(model_m)
model_b=BatchNormalization()(model_4)
model_d = Dropout(0.3)(model_b)
model_m = MaxPool2D(2,2)(model_d)

model_5 = Conv2D(32, (3,3), activation = 'relu')(model_m)
model_b=BatchNormalization()(model_5)
model_d = Dropout(0.3)(model_b)
model_m = MaxPool2D(2,2)(model_d)


model_flatten = Flatten()(model_m)                              
model_b=BatchNormalization()(model_flatten)


model_dense = Dense(256, activation='relu')(model_b)
model_b=BatchNormalization()(model_dense)
model_d = Dropout(0.4)(model_b)


# model_dense = Dense(512, activation= 'relu')(model_d)
# model_d = Dropout(0.5)(model_dense)


model_output = Dense(10, activation= 'softmax')(model_d)


model = Model (inputs = input0, outputs= (model_output))

model.summary()


model.save('./model/sample/cifar10/model_cifar10.h5') 

# 훈련 


from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])


modelpath = './model/sample/cifar10{epoch:02d} - {val_loss: .4f}.hdf5' 
checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)

model.fit(x_train,y_train, epochs= 20, batch_size= 120, validation_split= 0.25 ,callbacks= [early_stopping,checkpoint])



model.save_weights('./model/sample/cifar10/mnist_cifar10.h5')

# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('accuracy : ', acc)

'''

loss : 1.2326649198872117
accuracy :  0.6212000250816345

loss : 0.7033573728081548
accuracy :  0.7678999900817871


'''
