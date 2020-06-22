
from xgboost import XGBRegressor, plot_importance, XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, accuracy_score


import matplotlib.pyplot as plt
import numpy as np

dataset = load_breast_cancer()

# x = dataset.data
# y = dataset.target 

x, y = load_breast_cancer(return_X_y=True)

print(x.shape)  
print(y.shape) 

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 16)


model = XGBClassifier( n_estimators = 300 , learning_rate = 0.1)

model.fit(x_train, y_train)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds :  
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)     
                                                                          
    select_x_train = selection.transform(x_train)
    select_x_test  = selection.transform(x_test)
    
    selection_model = XGBClassifier(n_jobs=-1)

    
    selection_model.fit(select_x_train, y_train, eval_metric = ["logloss","rmse","mae"],  
                                                 eval_set = [(select_x_train,y_train), (select_x_test,y_test)], 
                                                 early_stopping_rounds = 20, verbose=0)
    
    y_pred = selection_model.predict(select_x_test)
    
    acc = accuracy_score(y_test,y_pred)  

    # print("R2 : ", r2_score)
    
    # results = selection_model.evals_result()
    
    print("Thresh=%.3f, n=%d, Acc: %.2f%%" %(thresh, select_x_train.shape[1], acc*100.0))
    
    
'''

Thresh=0.000, n=30, Acc: 100.00%
Thresh=0.001, n=29, Acc: 100.00%
Thresh=0.001, n=28, Acc: 100.00%
Thresh=0.001, n=27, Acc: 100.00%
Thresh=0.002, n=26, Acc: 100.00%
Thresh=0.003, n=25, Acc: 100.00%
Thresh=0.004, n=24, Acc: 100.00%
Thresh=0.004, n=23, Acc: 100.00%
Thresh=0.004, n=22, Acc: 100.00%
Thresh=0.004, n=21, Acc: 100.00%
Thresh=0.004, n=20, Acc: 100.00%
Thresh=0.006, n=19, Acc: 100.00%
Thresh=0.006, n=18, Acc: 100.00%
Thresh=0.006, n=17, Acc: 100.00%
Thresh=0.008, n=16, Acc: 100.00%
Thresh=0.009, n=15, Acc: 100.00%
Thresh=0.009, n=14, Acc: 100.00%
Thresh=0.010, n=13, Acc: 100.00%
Thresh=0.011, n=12, Acc: 100.00%
Thresh=0.012, n=11, Acc: 100.00%
Thresh=0.013, n=10, Acc: 100.00%
Thresh=0.014, n=9, Acc: 100.00%
Thresh=0.017, n=8, Acc: 100.00%
Thresh=0.020, n=7, Acc: 91.30%
Thresh=0.025, n=6, Acc: 95.65%
Thresh=0.054, n=5, Acc: 95.65%
Thresh=0.056, n=4, Acc: 86.96%
Thresh=0.071, n=3, Acc: 86.96%
Thresh=0.118, n=2, Acc: 100.00%
Thresh=0.507, n=1, Acc: 82.61%


'''