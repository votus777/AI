
from xgboost import XGBRegressor, plot_importance, XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, accuracy_score

import matplotlib.pyplot as plt

dataset = load_iris()

# x = dataset.data
# y = dataset.target 

x, y = load_iris(return_X_y=True)

print(x.shape)  
print(y.shape) 

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 16)


model = XGBClassifier( n_estimators = 300 , learning_rate = 0.1)

model.fit(x_train, y_train, verbose = True, eval_metric = ["mlogloss"],  
                                            eval_set = [(x_train,y_train), (x_test,y_test)], 
                                            early_stopping_rounds = 20)


 # eval_metric => rmse, mae, logloss, error, auc, roc 
 
results = model.evals_result()

# print("eval's result : ", results)


y_pred = model.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print("Acc score : %.2ff%%" %(acc*100.0))


epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)


fig, ax = plt.subplots()
ax.plot  (x_axis,  results['validation_0']['mlogloss'], label = 'Train')
ax.plot  (x_axis,  results['validation_0']['mlogloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

plt.show()

'''
[1]     validation_0-mlogloss:0.85385   validation_1-mlogloss:0.91171
[2]     validation_0-mlogloss:0.75921   validation_1-mlogloss:0.84289
[3]     validation_0-mlogloss:0.67910   validation_1-mlogloss:0.78654
[4]     validation_0-mlogloss:0.60943   validation_1-mlogloss:0.74021
[5]     validation_0-mlogloss:0.54885   validation_1-mlogloss:0.70155
[6]     validation_0-mlogloss:0.49561   validation_1-mlogloss:0.66979
[7]     validation_0-mlogloss:0.44906   validation_1-mlogloss:0.64367
[8]     validation_0-mlogloss:0.40801   validation_1-mlogloss:0.62206
[9]     validation_0-mlogloss:0.37068   validation_1-mlogloss:0.60413
[10]    validation_0-mlogloss:0.33755   validation_1-mlogloss:0.59014
[11]    validation_0-mlogloss:0.30806   validation_1-mlogloss:0.57934
[12]    validation_0-mlogloss:0.28175   validation_1-mlogloss:0.57130
[13]    validation_0-mlogloss:0.25825   validation_1-mlogloss:0.56568
[14]    validation_0-mlogloss:0.23721   validation_1-mlogloss:0.56217
[15]    validation_0-mlogloss:0.21834   validation_1-mlogloss:0.56065
[16]    validation_0-mlogloss:0.20140   validation_1-mlogloss:0.56076
[17]    validation_0-mlogloss:0.18622   validation_1-mlogloss:0.56231
[18]    validation_0-mlogloss:0.17254   validation_1-mlogloss:0.56510
[19]    validation_0-mlogloss:0.16019   validation_1-mlogloss:0.56898
[20]    validation_0-mlogloss:0.14903   validation_1-mlogloss:0.57399
[21]    validation_0-mlogloss:0.13886   validation_1-mlogloss:0.57987
[22]    validation_0-mlogloss:0.12974   validation_1-mlogloss:0.58648
[23]    validation_0-mlogloss:0.12140   validation_1-mlogloss:0.59377
[24]    validation_0-mlogloss:0.11385   validation_1-mlogloss:0.60160
[25]    validation_0-mlogloss:0.10699   validation_1-mlogloss:0.60989
[26]    validation_0-mlogloss:0.10076   validation_1-mlogloss:0.61858
[27]    validation_0-mlogloss:0.09510   validation_1-mlogloss:0.62760
[28]    validation_0-mlogloss:0.08993   validation_1-mlogloss:0.63713
[29]    validation_0-mlogloss:0.08526   validation_1-mlogloss:0.64646
[30]    validation_0-mlogloss:0.08096   validation_1-mlogloss:0.65134
[31]    validation_0-mlogloss:0.07656   validation_1-mlogloss:0.65668
[32]    validation_0-mlogloss:0.07253   validation_1-mlogloss:0.66205
[33]    validation_0-mlogloss:0.06881   validation_1-mlogloss:0.66714
[34]    validation_0-mlogloss:0.06537   validation_1-mlogloss:0.67258
[35]    validation_0-mlogloss:0.06220   validation_1-mlogloss:0.67799
Stopping. Best iteration:
[15]    validation_0-mlogloss:0.21834   validation_1-mlogloss:0.56065

Acc score : 83.33f%
'''