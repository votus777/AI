
'''
과적합 방지 
1. 훈련 데이터량을 늘린다. 
2. 피쳐 수를 줄인다.     -> 이걸 여기서 해보자   # feature engineering
3. regularization 
'''


from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt

dataset = load_boston()

x = dataset.data
y = dataset.target 

print(x.shape)  # (506, 13)
print(y.shape)  # (506,)

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
점수 :  0.9627799562644529
'''