from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV


import matplotlib.pyplot as plt

dataset = load_iris()

x = dataset.data
y = dataset.target 

print(x.shape) 
print(y.shape) 

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 14)



n_estimators = 10000
learning_rate = 0.001

colsample_bytree = 0.4 # ( 보통 0.6 ~ 0.9 사이 ) 개별 의사결정나무 모형에 사용될 변수갯수를 지정
colsample_bylevel = 0.4  # ( 이하동문)  

max_depth = 5
n_jobs = -1 

parameters = [ {"n_estimators" : [ 100, 200, 300], "learning_rate" : [ 0.001, 0.01, 0.1], "max_depth" : [4,5,6]},
               
               {"n_estimators" : [ 50, 200, 300], "learning_rate" : [ 0.001, 0.01, 0.1], 
                "max_depth" : [4,5,6], "colsample_bytree":[0.6, 0.9, 1], "gamma" : [0.1, 0.5, 0.9]}, 
               
               {"n_estimators" : [ 90, 200 ], "learning_rate" : [ 0.001, 0.01, 0.1],
                 "max_depth" : [4,5,6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9],"gamma" : [0.1, 0.5, 0.9]} 
               
              
              ]

xgb = XGBRegressor()


model = GridSearchCV(xgb, parameters, cv =5, n_jobs= -1 )

model.fit(x_train,y_train)







print(model.best_estimator_)
print('========================')
print(model.best_params_)

print('========================')
score = model.score(x_test,y_test)
print('점수 : ', score)

print('========================')
# plot_importance(model)    feature, plot 은 xgb 전용
plt.show()






'''

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0.1, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=50, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bytree': 0.9, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50}
========================
점수 :  0.9973630691427501
========================


'''