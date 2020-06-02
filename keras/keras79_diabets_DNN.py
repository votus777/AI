

from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.datasets  import load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt 
import numpy as np



diabetes = load_diabetes()
x, y = diabetes.data, diabetes.target


# print(x.shape)   # (442, 10)
# print(y.shape)   # (442, )


x = x[:, np.newaxis, 6]

# print(x[0])  # [[0.03807591 0.05068012 0.06169621 0.02187235]]
# print(x.shape)  # (442, 1, 4)

print(y[0])


# x = x.reshape(442,1)

'''
pca = PCA(n_components=1)
x = pca.fit_transform(x)
'''


#####################################
standard_scaler = StandardScaler()    

x = standard_scaler.fit_transform(x)


minmax_scaler = MinMaxScaler()

x = minmax_scaler.fit_transform(x)


# 겁나 삽질하고 있었는데 알고보니 scaler 적용이 안되고 있었다.. x = , y = 까먹지말자 
# y는 할 필요가 없댄다 다시 해보자 





print(y[0])

####################################






'''
If there are few data points per dimension, noise in the observations induces high variance:


Curse of High dimensions : 

추가된 변수가 실제 Y와 높은 관계가 있는 변수라면 적합에 도움이 되겠지만, 
추가된 변수가 반응변수Y와 실제 관계가 별로 없는 변수라면 오히려 이를 포함한 모델의 test error는 증가한다. 
이러한 noise feature들은 차원은 증가시키면서도 overfitting의 위험은 높이는 작용을 하게 된다

https://godongyoung.github.io/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/2018/02/07/ISL-Linear-Model-Selection-and-Regularization_ch6.html


This is an example of bias/variance tradeoff :  the larger the ridge alpha parameter, the higher the bias and the lower the variance.

Solutions : subset selection, Shrinkage, Dimension Reduction


'''



'''
transformer_PCA = PCA(n_components=1)  # PCA 차원 축소 
transformer_PCA.fit(x)

x= transformer_PCA.transform(x)
'''




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
   
    x,y, shuffle = True  , train_size = 0.8  
)


# print(x_train.shape)  # (353, 10)
# print(x_test.shape)  # (89, 10)




'''


print(diabets.DESCR)

**Data Set Characteristics:**

  :Number of Instances: 442   # 당뇨병 환자들 

  :Number of Attributes: First 10 columns are numeric predictive values

  :Target: Column 11 is a quantitative measure of disease progression one year after baseline  # 1년 뒤 측정한 당뇨병의 진행률

  :Attribute Information:
      - Age
      - Sex   -> 1 혹은 0 일텐데 값이 왜 소수점이 나오지..
      - Body mass index
      - Average blood pressure
      - S1
      - S2
      - S3
      - S4
      - S5
      - S6

Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times `n_samples` (i.e. the sum of squares of each column totals 1).

'''



# 모델  

model = Sequential()
model.add(Dense(16, activation='relu', input_dim = 1))
model.add(Dense(16, activation= 'relu')) 
model.add(Dense(32, activation= 'relu' )) 
model.add(Dropout(0.3))


model.add(Dense(64, activation= 'relu')) 
model.add(Dense(64, activation= 'relu')) 
model.add(Dense(64, activation= 'relu')) 
model.add(Dropout(0.3))


model.add(Dense(32, activation= 'relu')) 
model.add(Dense(16, activation= 'relu')) 
model.add(Dense(16, activation= 'relu')) 
model.add(Dropout(0.3))

model.add(Dense(1, activation= 'relu')) 


# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 10, mode ='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

hist = model.fit(x_train,y_train, epochs= 10000, batch_size= 1, validation_split= 0.2,  callbacks= [early_stopping])



# 평가 및 예측 


loss, mse = model.evaluate(x_test,y_test, batch_size=1)


print('loss :', loss)
print('mse : ', mse)



#________R2 구하기_____________________


y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)

print("R2 score : ", r2)

'''


'''