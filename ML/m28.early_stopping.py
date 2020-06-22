
from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target 

print(x.shape)  
print(y.shape) 

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 16)


model = XGBRegressor( n_estimators = 300 , learning_rate = 0.1)

model.fit(x_train, y_train, verbose = True, eval_metric = "rmse",  
                                            eval_set = [(x_train,y_train), (x_test,y_test)], 
                                            early_stopping_rounds = 20)


 # eval_metric => rmse, mae, logloss, error, auc, roc 
 
results = model.evals_result()

print("eval's result : ", results)


y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)

print("R2 score : %.2ff%%" %(r2*100.0))


'''

early stopping을 하게 되면 20번 patience가 끝났을 때의 값이 아니라 20번 돌기 전의 값을 가져온다 


'''

