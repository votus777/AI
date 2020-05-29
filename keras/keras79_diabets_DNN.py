
from keras.utils import np_utils
from keras.models import Sequential, Model 
from keras.layers import Input, Dense , LSTM, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn  import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 
import numpy as np

diabets = datasets.load_diabetes()


x = diabets.data
y = diabets.target

# print(x[0])   # (442, 10)
# print(y)   # (442, )

# print(np.std(y))     # np.mean(y) = 152,   표준편차 : 77.00574586945044  -> 일단  150 이상이면 1이라 둔다. 깊게는 생각 안함


# 데이터 설명에  data가  이미 표준화 되어있으므로 

transformer_PCA = PCA(n_components=5)  # PCA 차원 축소 
transformer_PCA.fit(x)

x = transformer_PCA.transform(x)


print(x.shape)   # (442,5)


y[y < 150] = 0
y[y >= 150] = 1

# print(y)   # 1,0 도배 


'''
print(diabets.DESCR)

**Data Set Characteristics:**

  :Number of Instances: 442   # 당뇨병 환자들 

  :Number of Attributes: First 10 columns are numeric predictive values

  :Target: Column 11 is a quantitative measure of disease progression one year after baseline  # 1년 뒤 측정한 당뇨병의 진행률

  :Attribute Information:
      - Age
      - Sex
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
model.add(Dense(50, activation='relu', input_dim = 5))
model.add(Dense(20, activation= 'sigmoid'))
model.add(Dense(20, activation= 'sigmoid'))


model.add(Dense(1, activation= 'sigmoid'))




# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 10, mode ='auto')

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

hist = model.fit(x,y, epochs= 10000, batch_size= 1, validation_split= 0.25,  callbacks= [early_stopping])



# 평가 및 예측 


loss, acc = model.evaluate(x,y, batch_size=1)


print('loss :', loss)
print('accuracy : ', acc)




'''

대충 값은 나온다

loss : 0.4625585946250218
accuracy :  0.7647058963775635

튜닝은 나중에 

'''