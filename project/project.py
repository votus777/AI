
# 전이학습을 이용한 사용자 분류 
# ResNet-100


from keras.datasets import cifar100, cifar10


from keras.applications import VGG16, VGG19, Xception, ResNet101
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D, Dropout
from keras.optimizers import Adam 



(x_train, y_train),(x_test,y_test) = cifar10.load_data()


x_train  = x_train.reshape(50000,32,32,3).astype('float32')/255.0  
x_test  = x_test.reshape(10000,32,32,3).astype('float32')/255.0  


resnet101 = ResNet101(input_shape = (32, 32, 3), include_top = False)



model = Sequential()
model.add(resnet101)
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


model.compile(optimizer=Adam(2e-4), loss = 'sparse_categorical_crossentropy', metrics=['acc'] )

model.fit(x_train,y_train, epochs= 30, batch_size= 500, validation_split= 0.2)




# 평가 및 예측 


loss, acc = model.evaluate(x_train,y_train)
val_loss, val_acc = model.evaluate(x_test, y_test)
  
print('loss :', loss)
print('accuracy : ', acc)
print('val_loss :', val_loss)
print('val_accuracy : ', val_acc)

