from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV


import matplotlib.pyplot as plt

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target 

print(x.shape)  
print(y.shape)  

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 18)


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
