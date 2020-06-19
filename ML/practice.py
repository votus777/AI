import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.datasets import load_boston



# dataset = load_boston()

# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)


x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 12)


model = XGBRegressor()
model.fit(x_train,y_train)

# score = model.score(x_test,y_test)

# print("R2 :", score)


thresholds = np.sort(model.feature_importances_)

print(thresholds)

for thresh in thresholds :  
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)      
                                                                          
    select_x_train = selection.transform(x_train)
    
    
    
    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)
    
    select_x_test  = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score  =  r2_score(y_test,y_pred)
    # print("R2 : ", r2_score)
    
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))