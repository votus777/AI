
from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt

dataset = load_iris()

# x = dataset.data
# y = dataset.target 

x, y = load_iris(return_X_y=True)

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 12)



n_estimators = 10000
learning_rate = 0.001

colsample_bytree = 0.4 # ( 보통 0.6 ~ 0.9 사이 ) 개별 의사결정나무 모형에 사용될 변수갯수를 지정
colsample_bylevel = 0.4  # ( 이하동문)  

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

점수 :  0.9987910519009804

'''