
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


x = np.array(diabets.data)
y = np.array(diabets.target)

# print(x[0])   # (442, 10)
# print(y)   # (442, )

# print(np.std(y))     # np.mean(y) = 152,   표준편차 : 77.00574586945044  -> 일단  150 이상이면 1이라 둔다. 깊게는 생각 안함 -> 그냥 회귀분석하자 


# 데이터 설명에  data가  이미 표준화 되어있으므로 



print(x)   # (442,10)
print(y)   #(442, )


transformer_Standard = StandardScaler()    
transformer_Standard.fit(x)

x = transformer_Standard.transform(x)


x.sort()




# print(x[0]) = [ 0.80050009  1.06548848  1.29708846  0.45983993 -0.92974581 -0.7320646  -0.91245053 -0.05449919  0.41855058 -0.37098854]
# print(x[0]) , x.sort()  = [-0.92974581 -0.91245053 -0.73206462 -0.37098854 -0.05449919  0.41855058 0.45983993  0.80050009  1.06548848  1.29708846]

x_n = x[ :, :4]   
x_p = x[ :, 5:]   


transformer_PCA = PCA(n_components=4)  # PCA 차원 축소 
transformer_PCA.fit(x_p)

x_p = transformer_PCA.transform(x_p)




print(x_n[0])      # [-0.92974581 -0.91245053 -0.73206462 -0.37098854]
print(x_p[0])      # [0.41855058 0.45983993 0.80050009 1.06548848 1.29708846]

print(x_n.shape)  # (442,4)
print(x_p.shape)  # (442,4)

'''
from sklearn.model_selection import train_test_split
x_n_train, x_p_train, x_n_test, x_p_test, y_train, y_test = train_test_split(
   
    x_n, x_p, y, shuffle = True  , train_size = 0.8  
)

'''


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

#model -------- 1
input1 = Input(shape=(4, ), name= 'input_n') 

dense1_1 = Dense(24, activation= 'relu', name= '1_1') (input1) 
dense1_2 = Dense(64, name = '1_2')(dense1_1)
dense1_3 = Dense(256,activation='relu', name = '1_3')(dense1_2)


#model -------- 2
input2 = Input(shape=(4, ), name = 'input_p') 

dense2_1 = Dense(24, activation= 'relu', name = '2_1')(input1) 
dense2_2 = Dense(64, name = '2_2')(dense2_1)
dense2_3 = Dense(256,activation='relu', name = '2_3')(dense2_2)


#이제 두 개의 모델을 엮어서 명시 

from keras.layers.merge import concatenate    #concatenate : 사슬 같이 잇다
merge1 = concatenate([dense1_3, dense2_3], name = 'merge') #파이썬에서 2개 이상은 무조건 list []

middle1 = Dense(128, activation= 'relu')(merge1)
middle1 = Dense(128, activation= 'relu')(merge1)
middle1 = Dense(128, activation= 'relu')(merge1)
middle1 = Dense(128, activation= 'relu')(merge1)
middle1 = Dense(128, activation= 'relu')(merge1)
middle1 = Dense(128, activation= 'relu')(merge1)
middle1 = Dense(128, activation= 'relu')(middle1)


################# output 모델 구성 ####################


output1 = Dense  (256, activation= 'relu',name = 'output_1')(middle1)
output1_2 = Dense (64, activation= 'relu',name = 'output_1_2')(output1)
output1_3 = Dense (32, activation= 'relu',name = 'output_1_3')(output1_2)
output1_4 = Dense (1, name = 'output_1_4')(output1_3)



model = Model (inputs = [input1, input2], outputs= (output1_4))


# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 50, mode ='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

hist = model.fit([x_n,x_p],y, epochs= 10000, batch_size= 1, validation_split= 0.2,  callbacks= [early_stopping])



# 평가 및 예측 


loss, mse = model.evaluate([x_n,x_p],y, batch_size=1)


print('loss :', loss)
print('mse : ', mse)


