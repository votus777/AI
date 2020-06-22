
'''

m30에 _eval2 와 eval3 만들것

1. eval 예 'loss' 와 다른 지표 1개 더 추가 
2. earlystopping 적용
3. plot 으로 그릴 것 

SelectfromModel 에 
1. 회귀                             m29_eval1_SFM
2. 이진 분류                        m29_eval2_SFM
3. 다중 분류                        m29_eval3_SFM


4. 결과는 주석으로 소스 하단에 표시 

5. m27 ~ 29까지 완벽 이해할 것 


'''


from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

dataset = load_boston()

x, y = load_boston(return_X_y=True)

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
