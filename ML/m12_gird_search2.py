
# from m12_gird_search.py

import pandas as pd

from sklearn.model_selection import train_test_split , KFold, GridSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC


# 1. 데이터 
iris = pd.read_csv('./data/csv/iris.csv', header = 0, )

x = iris.iloc[ :, 0 : 4 ]

y = iris.iloc[:,4]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 31)

parameters = [ 
    {"C" : [ 1, 10, 100, 1000], "kernel":["linear"]},
    {"C" : [ 1, 10, 100, 1000], "kernel":["rbf"],"gamma" : [0.001, 0.0001]},
    {"C" : [ 1, 10, 100, 1000], "kernel":["sigmoid"],"gamma" : [0.001, 0.0001]}    # 각 조합을 시도해본다  /C : 1 , Linear, gamma: 0.001/
    ]                                                                              #                     /C : 10, rbf , gamma : 0.0001/ 등등   





kfold = KFold(n_splits =5, shuffle = True)
model = Random   # CV - cross vailidation  

model.fit(x_train, y_train)  # -> test는 이미 분리되어 있고, train에서 CV 하겠다 


y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred))

'''
최종 정답률 :  0.9333333333333333 -> 그런데 뭐가 0.9333이 나왔는지 모른다 

'''

print( " 최적의 매게변수 : ", model.best_estimator_)
