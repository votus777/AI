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

# score = model.score(x_test,y_test)

# print("R2 :", score)


thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds :   # 전체 feature 수 만큼 모델을 돌리겠다  
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)     # 과제 1 : 또 다른 파라미터인 median은 무엇인가?  
                                                                          # 과제 2 : 이 함수에 girdsearch 적용시키기  -> parameter & feature 한방에 자동화 
    select_x_train = selection.transform(x_train)
    
    # print(select_x_train.shape) # 중요하지 않은 feature가 한 개씩 빠지는 걸 볼 수 있음 ( np.sort )
    
    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)
    
    select_x_test  = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score  =  r2_score(y_test,y_pred)
    # print("R2 : ", r2_score)
    
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    
'''
Thresh=0.001, n=13, R2: 95.61%
Thresh=0.002, n=12, R2: 95.72%
Thresh=0.010, n=11, R2: 95.63%
Thresh=0.011, n=10, R2: 94.69%
Thresh=0.014, n=9, R2: 95.75%
Thresh=0.014, n=8, R2: 95.83%
Thresh=0.019, n=7, R2: 93.97%
Thresh=0.037, n=6, R2: 95.14%
Thresh=0.051, n=5, R2: 91.60%
Thresh=0.054, n=4, R2: 95.74%
Thresh=0.056, n=3, R2: 96.34%   <-
Thresh=0.154, n=2, R2: 89.69%
Thresh=0.577, n=1, R2: 64.51%
중요도               

'''



'''

과제 

일요일 밤까지 

메일 제목 : " 누구누구 몇 등" 


1. 그리드 서치 엮기 및 소스 코드 보내기 
2. 데이콘 제출

3. 26_2,3 파일 만들기 


'''