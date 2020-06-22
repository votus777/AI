
from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score


import matplotlib.pyplot as plt
import numpy as np

dataset = load_boston()

# x = dataset.data
# y = dataset.target 

x, y = load_boston(return_X_y=True)

print(x.shape)  
print(y.shape) 

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 16)


model = XGBRegressor( n_estimators = 300 , learning_rate = 0.1)

model.fit(x_train, y_train)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds :  
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)     
                                                                          
    select_x_train = selection.transform(x_train)
    select_x_test  = selection.transform(x_test)
    
    selection_model = XGBRegressor(n_jobs=-1)

    
    selection_model.fit(select_x_train, y_train, eval_metric = ["logloss","rmse","mae"],  
                                                 eval_set = [(select_x_train,y_train), (select_x_test,y_test)], 
                                                 early_stopping_rounds = 20,verbose=0)
    
    y_pred = selection_model.predict(select_x_test)
    
    score  =  r2_score(y_test,y_pred)
    # print("R2 : ", r2_score)
    
    results = selection_model.evals_result()
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    
    
'''
Thresh=0.002, n=13, R2: 85.98%
Thresh=0.004, n=12, R2: 85.98%
Thresh=0.008, n=11, R2: 85.99%
Thresh=0.009, n=10, R2: 85.67%
Thresh=0.009, n=9, R2: 85.79%
Thresh=0.013, n=8, R2: 85.70%
Thresh=0.016, n=7, R2: 86.26%
Thresh=0.032, n=6, R2: 82.53%
Thresh=0.036, n=5, R2: 83.21%
Thresh=0.055, n=4, R2: 87.94%
Thresh=0.058, n=3, R2: 84.12%
Thresh=0.243, n=2, R2: 80.88%
Thresh=0.516, n=1, R2: 78.92%
'''