
# from m14_pipeline.py

# RandomizedSearchCV + pipeline

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier as rf

# 데이터 
iris = load_iris()

x = iris.data
y = iris.target

# keras 는 x,y 한방에 떙기고 sklearn은 이렇게 나눠서 가져온다 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 58)

# GridSearch/RandomSearch에서 사용할 매게 변수 
parameters =  [
    {'rf__max_leaf_nodes' : [10,100,150], 'rf__n_estimators' : [10,100,150,200]},
    {'rf__max_leaf_nodes' : [10,100,150], 'rf__n_estimators' : [10,100,150,200], 'rf__max_depth': [1,2]},
    {'rf__max_leaf_nodes' : [10,100,150], 'rf__n_estimators' : [10,100,150,200], 'rf__max_depth' : [1,2]}
]


# 0.22.1
# 모델

# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', rf())])  # model 과 전처리 방법을 선택  
# pipe = make_pipeline (MinMaxScaler(),rf() )  #-> 만약 안되면 이 라인으로 시도, 전처리기와 모델 이름을 파라미터 이름에 똑같이 넣어주어야 한다  'svc__C'


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
                ('rf',
                 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                        class_weight=None, criterion='gini',
                                        max_depth=None, max_features='auto',
                                        max_leaf_nodes=10, max_samples=None,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0,
                                        n_estimators=100, n_jobs=None,
                                        oob_score=False, random_state=None,
                                        verbose=0, warm_start=False))],
         verbose=False)


'''