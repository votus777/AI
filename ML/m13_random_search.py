
# from m01_selectModel.py

import pandas as pd

from sklearn.model_selection import train_test_split , KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

from sklearn.datasets import load_breast_cancer

# 1. 데이터 

breast_cancer = load_breast_cancer()


x = breast_cancer.data
y = breast_cancer.target

print(x.shape) # (569, 30)
print(y.shape)  # (569,)


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)


parameters = {"n_estimators" : [ 1, 50, 100], "max_depth":[1, 10, 50], 
     "max_features" :[1, 10, 30], "min_samples_split":[0.1, 0.5, 0.9]}  # -> gridsearch와 다르게 list 형식이 아니다  
                                                                          

 
classifier = RandomForestClassifier()

kfold = KFold(n_splits =10, shuffle = True)
model = RandomizedSearchCV(estimator = classifier, param_distributions=parameters, cv = kfold, verbose= 1, n_jobs= -1)   # CV - cross vailidation  


model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("최종 정답률 : ", model.score(x_test,y_pred))

'''

최종 정답률 :  1.0

'''
print( " 최적의 매게변수 : ", model.best_params_)
'''
 최적의 매게변수 :  {'n_estimators': 100, 'min_samples_split': 0.1, 'max_features': 1, 'max_depth': 50}

'''