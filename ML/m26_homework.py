import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score

from xgboost import XGBRegressor as xg
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.datasets import load_boston

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import itertools



# 데이터
x, y = load_boston(return_X_y=True)


# 데이터  split
x_train,x_test, y_train, y_test = train_test_split(x, y, train_size = 0.96, shuffle = 'True', random_state = 12)

# 모델 
model = xg()
model.fit(x_train,y_train)

# Feature Importance 추출 
thresholds = np.sort(model.feature_importances_)

print(thresholds)


# Select from Model 
for thresh in thresholds :     
    
    
    parameters = [ {"n_estimators" : [ 100, 200, 300], "learning_rate" : [ 0.001, 0.01, 0.1], "max_depth" : [4,5,6]},
               
               {"n_estimators" : [ 50, 200, 300], "learning_rate" : [ 0.001, 0.01, 0.1], 
                "max_depth" : [4,5,6], "colsample_bytree":[0.6, 0.9, 1], "gamma" : [0.1, 0.5, 0.9]}, 
               
               {"n_estimators" : [ 90, 200 ], "learning_rate" : [ 0.001, 0.01, 0.1],
                 "max_depth" : [4,5,6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9],"gamma" : [0.1, 0.5, 0.9]} 
               
              
              ]
  
  
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)                                                                  
    select_x_train = selection.transform(x_train)    # feature 값을 갱신하고  
    
    
    
    xgb = xg()             # 이 모델로노미콘
    
    grid = GridSearchCV(xgb, parameters, cv =5, n_jobs= -1 ) 
     # GridSearch를 해준다, 늘 하던대로 아무 생각없이 model로 변수 설정하면 위에 feature 추출할 때 쓴 model=xg()랑 위에 line54 부분에서 충돌난다, 조심하자
    
        
    grid.fit(select_x_train, y_train)   # 그 뒤로는 똑같다 
    
    select_x_test  = selection.transform(x_test)
    y_pred = grid.predict(select_x_test)
    
    score  =  r2_score(y_test,y_pred)
    
    
    print(grid.best_estimator_)
    print('========================')
    print(grid.best_params_)

    print('========================')
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    

'''
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0.1, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=200, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 1, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
========================
Thresh=0.001, n=13, R2: 96.06%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0.9, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=200, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 1, 'gamma': 0.9, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
========================
Thresh=0.002, n=12, R2: 95.86%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0.5, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=200, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 0.9, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}
========================
Thresh=0.010, n=11, R2: 96.92%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0.1, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 0.9, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 90}
========================
Thresh=0.011, n=10, R2: 96.13%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0.5, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.7, 'colsample_bytree': 0.9, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 90}
========================
Thresh=0.014, n=9, R2: 96.13%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0.9, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 1, 'gamma': 0.9, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 90}
========================
Thresh=0.014, n=8, R2: 96.84%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0.1, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.7, 'colsample_bytree': 0.9, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 90}
========================
Thresh=0.019, n=7, R2: 95.91%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,
             colsample_bynode=1, colsample_bytree=1, gamma=0.5, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.9, 'colsample_bytree': 1, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 90}
========================
Thresh=0.037, n=6, R2: 93.97%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0.9, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 1, 'gamma': 0.9, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 90}
========================
Thresh=0.051, n=5, R2: 95.14%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0.5, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 0.9, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 90}
========================
Thresh=0.054, n=4, R2: 96.22%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0.9, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=90, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bylevel': 0.6, 'colsample_bytree': 0.9, 'gamma': 0.9, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 90}
========================
Thresh=0.056, n=3, R2: 94.73%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.5, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=50, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bytree': 1, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50}
========================
Thresh=0.154, n=2, R2: 93.10%
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0.5, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.01, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=300, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
========================
{'colsample_bytree': 0.6, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300}
========================
Thresh=0.577, n=1, R2: 79.61%



'''

