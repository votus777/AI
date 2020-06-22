import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.datasets import load_iris


x, y = load_iris(return_X_y=True)


x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 12)


model = XGBClassifier()
model.fit(x_train,y_train)

# score = model.score(x_test,y_test)

# print("R2 :", score)


thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds :   
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)     
                                                                          
    select_x_train = selection.transform(x_train)
    
   
    
    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)
    
    select_x_test  = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    acc = accuracy_score(y_test,y_pred)  
    
    
    print("Thresh=%.3f, n=%d, Acc: %.2f%%" %(thresh, select_x_train.shape[1], acc*100.0))
 

'''
Thresh=0.009, n=4, Acc: 100.00%
Thresh=0.018, n=3, Acc: 100.00%
Thresh=0.298, n=2, Acc: 100.00%
Thresh=0.676, n=1, Acc: 100.00%

'''




