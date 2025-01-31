
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn  import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 
import numpy as np

breast_cancer = datasets.load_breast_cancer()


x = breast_cancer.data
y = breast_cancer.target

print(x.shape)   #(569, 30)  # 소수점의 향연 
print(y.shape)   #(569,)    # 1과 0의 향연


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)


'''

print(breast_cancer.DESCR)

**Data Set Characteristics:**

    :Number of Instances: 569

    :Number of Attributes: 30 numeric, predictive attributes and the class

    :Attribute Information:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)

        The mean, standard error, and "worst" or largest (mean of the three
        largest values) of these features were computed for each image,
        resulting in 30 features.  For instance, field 3 is Mean Radius, field
        13 is Radius SE, field 23 is Worst Radius.

        - class:
                - WDBC-Malignant
                - WDBC-Benign

    :Summary Statistics:

    ===================================== ====== ======
                                           Min    Max
    ===================================== ====== ======
    radius (mean):                        6.981  28.11
    texture (mean):                       9.71   39.28
    perimeter (mean):                     43.79  188.5
    area (mean):                          143.5  2501.0
    smoothness (mean):                    0.053  0.163
    compactness (mean):                   0.019  0.345
    concavity (mean):                     0.0    0.427
    concave points (mean):                0.0    0.201
    symmetry (mean):                      0.106  0.304
    fractal dimension (mean):             0.05   0.097
    radius (standard error):              0.112  2.873
    texture (standard error):             0.36   4.885
    perimeter (standard error):           0.757  21.98
    area (standard error):                6.802  542.2
    smoothness (standard error):          0.002  0.031
    compactness (standard error):         0.002  0.135
    concavity (standard error):           0.0    0.396
    concave points (standard error):      0.0    0.053
    symmetry (standard error):            0.008  0.079
    fractal dimension (standard error):   0.001  0.03
    radius (worst):                       7.93   36.04
    texture (worst):                      12.02  49.54
    perimeter (worst):                    50.41  251.2
    area (worst):                         185.2  4254.0
    smoothness (worst):                   0.071  0.223
    compactness (worst):                  0.027  1.058
    concavity (worst):                    0.0    1.252
    concave points (worst):               0.0    0.291
    symmetry (worst):                     0.156  0.664
    fractal dimension (worst):            0.055  0.208
    ===================================== ====== ======

    :Missing Attribute Values: None


'''



# 모델  

model = Sequential()
model.add(Dense(16, activation = 'relu', input_dim = 30))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.2))


model.add(Dense(32, activation= 'sigmoid'))
model.add(Dense(16, activation= 'sigmoid'))
model.add(Dense(16, activation= 'sigmoid'))




model.add(Dense(1, activation= 'sigmoid'))

model.save('./model/sample/cancer/model_cancer.h5') 



# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping( monitor='loss', patience= 50, mode ='auto')

modelpath = './model/sample/cancer{epoch:02d} - {val_loss: .4f}.hdf5' 

checkpoint = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only = True, save_weights_only= False, verbose=1)


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

hist = model.fit(x,y, epochs= 10000, batch_size= 1, validation_split= 0.25,  callbacks= [early_stopping, checkpoint])


model.save_weights('./model/sample/cancer/weights_cancer.h5')



# 평가 및 예측 


loss, acc = model.evaluate(x_test,y_test, batch_size=1)

  
print('loss :', loss)
print('acc : ', acc)

'''

loss : 0.16481326343164893
acc :  0.9473684430122375

'''