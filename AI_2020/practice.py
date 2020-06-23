import numpy as np
import pandas as pd

from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
# train = pd.read_csv('/tf/notebooks/train.csv', header = 0, index_col = 0)
# val = pd.read_csv('/tf/notebooks/validate.csv', header = 0, index_col = 0)
# test = pd.read_csv('/tf/notebooks/test.csv', header = 0, index_col = 0)

# train.shape (2896, 36)
# val.shape (721, 36)
# test.shape (721, 36)
# 1열 : 시간(0~23시),   2 ~ 36열 : 도로 

 #split 함수를 사용할까 ( 24시간 단위 )


 
def col_out(data):
    x_col = []
    y_col = []
    
    for i in range(data.shape[1]):
        x_col = data[: -1, i]
        y_col = train[24: , i]
        
        x = x_col
        y = y_col
        
        
        print("x.shape : ", x.shape)  # (2869, 36)
        print("y.shape : ", y.shape)  # (2869,)

# 전날 24시간 데이터로 다음 날 1시간 예측
# print(x[-1])
# print(y[0])

        x_train, x_test, y_train, y_test = train_test_split(
            x,y, train_size = 0.9, shuffle=False)


        print(x_train.shape)   # (2582, 36)
        print(x_test.shape)   # (287, 36)
        print(y_test.shape)  # (287, )


# x_train = x_train.reshape(2583,24,36)
# x_test = x_test.reshape(288,24,36)


        model = XGBRegressor(max_depth=7, learning_rate= 0.01, 
                            n_estimators=1200, colsample_bytree = 0.7, colsample_bylevel = 0.7, gamma = 0.15 ,n_jobs = -1)

        model.fit(x_train, y_train)

# model.compile(loss = 'mse', optimizer='adam', metrics = ['mse'])
# model.fit(x_train,y_train,epochs= 10, batch_size= 50, validation_split= 0.2)


        y_predict = model.predict(x_test)
        print('y_prdict : ', y_predict.shape)  # y_prdict :  (287,)
        print('y_test : ',y_test.shape)   #                (287, )


        print( y_predict)
        print('========================')
        print( y_test)
        print('========================')

        score = r2_score(y_test, y_predict)
        print('점수 : ', score)
        print('========================')






''' 
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten 

from xgboost import XGBRegressor, plot_importance

train = pd.read_csv('/tf/notebooks/train.csv', header = 1, index_col = 0)
val = pd.read_csv('/tf/notebooks/validate.csv', header = 1, index_col = 0)
test = pd.read_csv('/tf/notebooks/test.csv', header = 1, index_col = 0)

print("train.shape1 : ",train.shape)
print(val.shape)
print(test.shape)
 

train = train.interpolate(method = 'linear')  # 보간법  // 선형보간 
train = np.array(train)



# aindex = np.where(train == 38243 )
# print(aindex)  (array([2125]), array([1])), 2125, 2126 제외해야함 

train = np.delete(train,[2125,2126], axis=0)
print("train.shape2 : ",train.shape)

train[882] = [18,358700,110452,10507,26154,10729,6897,131517,7440,17017,10225,
              74782, 19115,28936,18464,38921,56762,10729,28943,75311,
              103770,77902,14450,14854,9762,169320,9122,15789,48820,
              23663,5424,23899,5718,23996,14877,24387]  # 바로 전날 값 가져오기 



def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i: (i + size)]
        aaa.append([item for item in subset])  
    print(type(aaa))
    return np.array(aaa)


dataset = split_x(train,24)
print("=============================")
# print(dataset[0])
print(dataset.shape) #(2872, 24, 36)

x = dataset[ : -1, 1] # 35개 도로에서 한개를 뽑는다 
y = train[24: , 1]

print("x.shape : ", x.shape)  # (2869, 36)
print("y.shape : ", y.shape)  # (2869,)

# 전날 24시간 데이터로 다음 날 1시간 예측
# print(x[-1])
# print(y[0])

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.9, shuffle=False)


print(x_train.shape)   # (2582, 36)
print(x_test.shape)   # (287, 36)
print(y_test.shape)  # (287, )


# x_train = x_train.reshape(2583,24,36)
# x_test = x_test.reshape(288,24,36)


model = XGBRegressor(max_depth=7, learning_rate= 0.01, 
                            n_estimators=1200, colsample_bytree = 0.7, colsample_bylevel = 0.7, gamma = 0.15 ,n_jobs = -1)

model.fit(x_train, y_train)

# model.compile(loss = 'mse', optimizer='adam', metrics = ['mse'])
# model.fit(x_train,y_train,epochs= 10, batch_size= 50, validation_split= 0.2)


y_predict = model.predict(x_test)
print('y_prdict : ', y_predict.shape)  # y_prdict :  (287,)
print('y_test : ',y_test.shape)   #                (287, )


print( y_predict)
print('========================')
print( y_test)
print('========================')

score = r2_score(y_test, y_predict)
print('점수 : ', score)
print('========================')

'''