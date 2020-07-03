
from keras.datasets import cifar100, cifar10


from keras.applications import VGG16, VGG19, Xception, ResNet101
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D, Dropout
from keras.optimizers import Adam 



(x_train, y_train),(x_test,y_test) = cifar10.load_data()


vgg16 = VGG16(input_shape = (32, 32, 3), include_top = False)   # 현재 default input =   (None, 224, 224, 3)  


x_train  = x_train.reshape(50000,32,32,3).astype('float32')/255.0  
x_test  = x_test.reshape(10000,32,32,3).astype('float32')/255.0  



vgg16 = VGG16()


model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()




model.compile(optimizer=Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics=['acc'] )

model.fit(x_train,y_train, epochs= 20, batch_size= 240, validation_split= 0.2)




# 평가 및 예측 


loss, acc = model.evaluate(x_train,y_train)
val_loss, val_acc = model.evaluate(x_test, y_test)
  
print('loss :', loss)
print('accuracy : ', acc)
print('val_loss :', loss)
print('val_accuracy : ', acc)

