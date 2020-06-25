import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

from statsmodels.tsa.arima_model import ARIMA


from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten 

from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor, LGBMClassifier 

train = pd.read_csv('./AI_2020/train.csv', header = 1, index_col = [0,1])
val = pd.read_csv('./AI_2020/val.csv', header = 1, index_col = [0,1])
test = pd.read_csv('./AI_2020/test.csv', header = 1, index_col = [0,1])

# print("val.shape : ",val.shape)   # (720, 36)
# print("test.shape : ",test.shape) # (720, 36)
 

train = np.array(train)
val = np.array(val)
test = np.array(test)



train[882] = np.array([ 308024, 87728, 8129, 21055, 8147, 5623, 115624, 5936, 13350, 8277, 
                        60908, 15114, 24214, 14920, 31827, 46209, 8595, 23577, 61015,
                        89910, 58269, 11297, 11059, 7054, 151508, 7251, 13648, 42689, 21848, 
                        4529, 19072, 4442, 19843, 11828, 19147])  # 2월 6일 18시 결측치 보간 

val[677] = np.array([ 131437,  35580,   2364,   6592,   2042,  1578, 50210,  2042,  4203,
                    2776, 18006,  5130,  9392,  4262,  8832, 21651,  4944, 13550,
                    26638, 40919, 20522,  2921,  3544,  3509, 78091,  2088,  7926,
                    19284, 10925,  1496,  6172,   750,  7477,  4876,  7578])  # 5월 15일 05시 결측치 보간 

val = val[323 : , : ] 

print(val)

def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1 ) :
        subset = seq[ i+11: (i + size)]
        aaa.append([item for item in subset])  
    print(type(aaa))
    return np.array(aaa)


train_dataset = split_x(train,35)
print("=============================")
print("train_dataset.shape : ", train_dataset.shape) #(2861, 24, 35)
print("=============================")

val_dataset = split_x(val,35)
print("val_dataset.shape : ", val_dataset.shape)  #(363, 24, 35)



def col_out(data,train,val_data,val):
    pred = np.array([])
    x_col = []
    y_col = []
    
   

    for i in range(data.shape[1]):
        
    
        
        x_col = data[: -1, i]
        y_col = train[24: , i+1]
        
        x_val_col = val_dataset[ : -1, i]
        
        x = x_col
        y = y_col

        x_val = x_val_col
        
        x_train, x_test, y_train, y_test = train_test_split(
            x,y, train_size = 0.8, shuffle=True, random_state = 25)


        model = LGBMRegressor(learning_rate= 0.01, n_estimators=1000, 
                              colsample_bytree = 0.9, n_jobs = -1, objective = 'regression',  max_bin =90, num_leaves=30, random_state=5   )
        # model = XGBRegressor(learning_rate= 0.01, n_estimators=1000, 
        #                       colsample_bytree = 0.9, n_jobs = -1, objective = 'reg:squarederror',
        #                       gpu_id=0, tree_method='gpu_hist', eval_metric='rmse', gamma = 0.01)

        model.fit(x_train, y_train)
       
        y_predict = model.predict(x_test)
        
    

        test = model.predict(x_val)
      
        print('========================')
        score = r2_score(y_test, y_predict)
        print('i=',i+1, ' 번째 도로의 점수 : ', score)
        print('========================')
        
        # saved_model = pickle.dumps(model)
        # joblib.dump(model, 'lgbm.pkl')
        
        pred = np.append(pred,test, axis=0)
        print(pred.shape)
        
        
     
    return np.array(pred)


     

a = col_out(train_dataset,train,val_dataset,val)
a = a.reshape(360,35)

index = [1,2,6,7,8,20,25,26,30,32]

pred = np.delete(a, index, axis=1)
print("result : ", pred.shape)

test_csv = pd.DataFrame(pred)
test_csv.to_csv('./AI_2020/predict_lgbm.csv', index=False, header=0)

# 별도의 인덱스 또는 컬럼명을 배제한 (360, 25) 형태의 csv 

