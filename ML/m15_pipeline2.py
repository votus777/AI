
# from m14_pipeline.py

# RandomizedSearchCV + pipeline

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 데이터 
iris = load_iris()

x = iris.data
y = iris.target

# keras 는 x,y 한방에 떙기고 sklearn은 이렇게 나눠서 가져온다 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 42)

# GridSearch/RandomSearch에서 사용할 매게 변수 
parameters =  [
    {'svc__C' : [1,10,100,1000], 'svc__kernel' : ['linear']},
    {'svc__C' : [1,10,100,1000], 'svc__kernel' : ['rbf'], 'svc__gamma': [0.001, 0.0001]},
    {'svc__C' : [1,10,100,1000], 'svc__kernel' : ['sigmoid'], 'svc__gamma' : [0.001, 0.0001]}
]

# parameters =  [
#     {"C" : [1,10,100,1000], "kernel" : ['linear']},
#     {"C" : [1,10,100,1000], "kernel" : ['rbf'], 'gamma': [0.001, 0.0001]},
#     {"C" : [1,10,100,1000], "kernel" : ['sigmoid'], 'gamma' : [0.001, 0.0001]}
# ]

#
# 0.22.1
# 모델

# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])  # model 과 전처리 방법을 선택  
pipe = make_pipeline (MinMaxScaler(), SVC() )  #-> 만약 안되면 이 라인으로 시도, 전처리기와 모델 이름을 파라미터 이름에 똑같이 넣어주어야 한다  'svc__C'


model = RandomizedSearchCV(pipe, parameters, cv=5)


# 훈련
model.fit(x_train,y_train)

# 평가, 예측
acc = model.score(x_test,y_test)

print("acc : " , acc) 
print("최적의 매게 변수 = ", model.best_estimator_)

'''

acc :  1.0
최적의 매게 변수 =  Pipeline(memory=None,
         steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                ('svm',
                 SVC(C=1000, break_ties=False, cache_size=200,
                     class_weight=None, coef0=0.0,
                     decision_function_shape='ovr', degree=3, gamma=0.0001,
                     kernel='rbf', max_iter=-1, probability=False,
                     random_state=None, shrinking=True, tol=0.001,
                     verbose=False))],
         verbose=False)


'''

'''
sklearn 버전 확인 
("sklearn : " , sklearn.__version__)

'''