
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

