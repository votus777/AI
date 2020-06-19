
from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target 

print(x.shape)  
print(y.shape) 

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 16)



n_estimators = 10000
learning_rate = 0.01

colsample_bytree = 0.97 # ( 보통 0.6 ~ 0.9 사이 ) 개별 의사결정나무 모형에 사용될 변수갯수를 지정
colsample_bylevel = 0.97  # ( 이하동문)  

max_depth = 5
n_jobs = -1 

model = XGBRegressor(max_depth=max_depth, learning_rate= learning_rate, 
                            n_estimators=n_estimators, n_jobs = n_jobs, 
                            corsample_bytree = colsample_bytree,    
                            colsample_bylevel= colsample_bylevel)

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print('점수 : ', score)
print('========================')
print(model.feature_importances_)


# plot_importance(model)
# plt.show()

'''
점수 :  0.9860062247963773

'''