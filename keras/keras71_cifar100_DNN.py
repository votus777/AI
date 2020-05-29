


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

x_train  = x_train.reshape(50000,3072).astype('float32')/255.0  
x_test  = x_test.reshape(10000,3072).astype('float32')/255.0  


# ____________모델 구성____________________



model= Sequential()

input1 = Input(shape=(3072,), name= 'input_1') 


dense1 = Dense  (216, activation = 'relu', name = 'output_1')(input1)
batch_1 = BatchNormalization()(dense1)
dropout1 = Dropout(0.25)(batch_1)


dense1_2 = Dense (512, activation = 'relu',  name = 'output_1_2')(dropout1)
batch_2 = BatchNormalization()(dense1_2)
dropout2 = Dropout(0.25)(batch_2)


dense1_3 = Dense (512, activation = 'relu', name = 'output_1_3')(dropout2)
batch_3 = BatchNormalization()(dense1_3)
dropout3 = Dropout(0.25)(batch_3)

dense1_4 = Dense (512, activation = 'relu', name = 'output_1_4')(dropout3)
batch_4 = BatchNormalization()(dense1_4)
dropout4 = Dropout(0.25)(batch_4)


dense1_5 = Dense (100, activation= 'softmax' , name = 'output_1_5')(batch_4)


model = Model (inputs = input1, outputs= (dense1_5))


model.summary()



# 훈련 



early_stopping = EarlyStopping( monitor='loss', patience= 100, mode ='auto')

modelpath = './model/{epoch:02d} - {val_loss: .4f}.hdf5' 

checkpoint = ModelCheckpoint(filepath= modelpath, monitor= ' val_loss', save_best_only = True, mode = 'auto')

tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph= True, write_images= True)

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])
hist = model.fit(x_train,y_train, epochs= 10, batch_size= 250, validation_split= 0.25 ,callbacks= [early_stopping,checkpoint,tb_hist])



# 평가 및 예측 


loss, acc = model.evaluate(x_train,y_train, batch_size=1)
val_loss, val_acc = model.evaluate(x_test, y_test, batch_size= 1)
  
print('loss :', loss)
print('accuracy : ', acc)

