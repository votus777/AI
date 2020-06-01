


from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

# 데이터 

(x_train, y_train),(x_test,y_test) = cifar100.load_data()


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)  # (50000, 100) -> one hot encoding 

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

model_4 = Conv2D(128, (5,5), activation = 'relu', padding= 'same' )(model_m)
model_b=BatchNormalization()(model_4)
model_d = Dropout(0.3)(model_b)
model_m = MaxPool2D(2,2)(model_d)

model_5 = Conv2D(64, (3,3), activation = 'relu', padding= 'same')(model_m)
model_b=BatchNormalization()(model_5)
model_d = Dropout(0.3)(model_b)
model_m = MaxPool2D(2,2)(model_d)


model_7 = Conv2D(32, (3,3), activation = 'relu', padding= 'same')(model_m)
model_b=BatchNormalization()(model_7)
model_d = Dropout(0.3)(model_b)
model_m = MaxPool2D(2,2)(model_d)




model_flatten = Flatten()(model_m)                              
model_b=BatchNormalization()(model_flatten)


model_dense = Dense(512, activation='relu')(model_b)
model_b=BatchNormalization()(model_dense)
model_d = Dropout(0.5)(model_b)

model_output = Dense(100, activation= 'softmax')(model_d)


model = Model (inputs = input0, outputs= (model_output))

model.summary()

# 훈련 



early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

modelpath = './model/{epoch:02d}-{val_loss: .4f}.hdf5' # 02d : 두자리 정수,  .4f : 소수점 아래 4자리 까지 float 실수

checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, mode = 'auto')

tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph= True, write_images= True)

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])
hist = model.fit(x_train,y_train, epochs= 20, batch_size= 240, validation_split= 0.2 , callbacks= [checkpoint, tb_hist])




# 평가 및 예측 


loss, acc = model.evaluate(x_train,y_train, batch_size=1)
val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  
print('loss :', loss)
print('accuracy : ', acc)



# 시각화


plt.figure(figsize= (10,6))



plt.subplot(2, 1, 1)    

plt.plot(hist.history['loss'] , marker = '.', c = 'red', label = 'loss')  
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')  
plt.grid()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])   
plt.show()



plt.subplot(2, 1, 2)   

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['acc', 'val_acc'])
plt.show()
