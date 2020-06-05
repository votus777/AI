
# from m01_selectModel.py

import pandas as pd

from sklearn.model_selection import train_test_split , KFold, GridSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from keras.datasets import cifar10, mnist
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


parameters = [ 
    {"n_estimators" : [ 1, 50, 100], "max_depth":[1, 10, 50], 
     "max_features" :[1, 10, 30], "min_samples_split":[0.1, 0.5, 0.9], 
     "warm_start":[True,False]}
    ]                                                                       

 

kfold = KFold(n_splits =10, shuffle = True)
model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold, verbose= 1, n_jobs= -1)   # CV - cross vailidation  

model.fit(x_train, y_train) 


y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred))







'''

최종 정답률 :  0.9824561403508771

'''


print( " 최적의 매게변수 : ", model.best_estimator_)


'''
 최적의 매게변수 :  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features=1, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=0.1,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

          "n_estimators" : [ 50 ], "max_depth":[ 10 ], "max_features" :[ 1 ], "min_samples_split":[ 0.1 ]


'''