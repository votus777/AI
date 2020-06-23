from lightgbm import LGBMRegressor, LGBMClassifier 

from xgboost import XGBRegressor, plot_importance  
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# 회귀 모델
x, y = load_boston(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model =LGBMRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= False, eval_metric= ['logloss', 'rmse'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    r2 = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thres, select_x_train.shape[1], r2*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # model.save_model("./model/xgb_save/boston_thresh=%.3f-r2=%.2f.model"%(thres, r2))
    model.save_model("./model/sample/boston/boston_rmse=%.3f-r2=%.2f.model"%(rmse, r2))
