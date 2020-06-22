import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBRegressor as xg
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.datasets import load_boston

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import itertools



# 데이터
x, y = load_boston(return_X_y=True)


# 데이터  split
x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 12)

# 모델 
model = xg()
model.fit(x_train,y_train)

# Feature Importance 추출 
thresholds = np.sort(model.feature_importances_)

print(thresholds)

import time
start = time.time()


import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.datasets import load_boston



# dataset = load_boston()

# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)


x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 12)


model = XGBRegressor()
model.fit(x_train,y_train)


thresholds = np.sort(model.feature_importances_)

print(thresholds)

import time
start = time.time()

for thresh in thresholds :  
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)     
                                                                          
    select_x_train = selection.transform(x_train)
    
   
    
    selection_model = XGBRegressor(n_estimators=1000, n_jobs=2)
    selection_model.fit(select_x_train, y_train)
    
    select_x_test  = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score  =  r2_score(y_test,y_pred)
    # print("R2 : ", r2_score)
    
    
    # print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    
end = time.time() - start



start2 = time.time()

for thresh in thresholds :  
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)     
                                                                          
    select_x_train = selection.transform(x_train)
    
   
    
    selection_model = XGBRegressor(n_jobs=6,n_estimators=1000 )
    selection_model.fit(select_x_train, y_train)
    
    select_x_test  = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score  =  r2_score(y_test,y_pred)
    # print("R2 : ", r2_score)
    
    
    # print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    


end2 = time.time() - start2

print("총 걸린 시간 : ", end) # 단위 : 초 
print("총 걸린 시간 feat n_jobs= -1 : ", end2) 

# XG에서 n_jobs 코어 갯수로 넣어준다 
# n_jobs= -1 로 넣을 경우 스레드 12개로 잡혀서 제대로 실행이 되지 않는 듯 

