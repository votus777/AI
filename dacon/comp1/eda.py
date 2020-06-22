import pandas as pd
import numpy as np
import pywt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputRegressor
import optuna
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_absolute_error
# import shap

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor as xg

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import warnings ; warnings.filterwarnings('ignore')
import time
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.metrics import mean_absolute_error


train=pd.read_csv('./data/dacon/comp1/train.csv', index_col='id')
test=pd.read_csv( './data/dacon/comp1/test.csv', index_col='id')
submission=pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col='id')




feature_names=list(test)
target_names=list(submission)

Xtrain = train[feature_names]
Xtest = test[feature_names]

bx = np.array(Xtrain)   # src
(ca, cd) = pywt.dwt(bx, "haar")
cat = pywt.threshold(ca, np.std(ca), mode="hard")
cdt = pywt.threshold(cd, np.std(cd), mode="hard")
x = pywt.idwt(cat, cdt, "haar")

bt = np.array(Xtest)   # test
(ca, cd) = pywt.dwt(bt, "haar")
cat = pywt.threshold(ca, np.std(ca), mode="hard")
cdt = pywt.threshold(cd, np.std(cd), mode="hard")
tx = pywt.idwt(cat, cdt, "haar")

Ytrain=train[target_names]
Ytrain1=Ytrain['hhb']
Ytrain2=Ytrain['hbo2']
Ytrain3=Ytrain['ca']
Ytrain4=Ytrain['na']


base_params={"n_estimators" : 500, "learning_rate" :  0.1,
                 "max_depth" : 4, "colsample_bytree":0.6, "colsample_bylevel" :0.6,"gamma" : 0.1, "n_jobs" : -1,
                 "objective":'reg:squarederror',"random_state":1}
                         
                               

base_model= xg(n_estimators = 1200, learning_rate=  0.1,
                 max_depth= 4, colsample_bytree=0.8, colsample_bylevel=0.8,gamma= 0.1, n_jobs= -1,
                 objective='reg:squarederror', random_state=31)

multi_model=MultiOutputRegressor(base_model)
  

def model_scoring_cv(model, x, y, cv=5):
    start=time.time()
    score=-cross_val_score(model, x, y, cv=cv, scoring='neg_mean_absolute_error').mean()
    stop=time.time()
    print(f"Validation Time : {round(stop-start, 3)} sec")
    return score


src_list=['650_src', '660_src', '670_src', '680_src', '690_src', '700_src', '710_src', '720_src', '730_src', 
          '740_src', '750_src', '760_src', '770_src', '780_src', '790_src', '800_src', '810_src', '820_src', 
          '830_src', '840_src', '850_src', '860_src', '870_src', '880_src', '890_src', '900_src', '910_src', 
          '920_src', '930_src', '940_src', '950_src', '960_src', '970_src', '980_src', '990_src']

dst_list=['650_dst', '660_dst', '670_dst', '680_dst', '690_dst', '700_dst', '710_dst', '720_dst', '730_dst', 
          '740_dst', '750_dst', '760_dst', '770_dst', '780_dst', '790_dst', '800_dst', '810_dst', '820_dst', 
          '830_dst', '840_dst', '850_dst', '860_dst', '870_dst', '880_dst', '890_dst', '900_dst', '910_dst', 
          '920_dst', '930_dst', '940_dst', '950_dst', '960_dst', '970_dst', '980_dst', '990_dst']

# model_scoring_cv(multi_model, Xtrain.fillna(-1), Ytrain)


alpha=Xtrain[dst_list]
beta=Xtest[dst_list]

for i in tqdm(Xtrain.index):
    alpha.loc[i] = alpha.loc[i].interpolate()
    
for i in tqdm(Xtest.index):
    beta.loc[i] = beta.loc[i].interpolate()

alpha.loc[alpha['900_dst'].isnull(),'900_dst']=alpha.loc[alpha['900_dst'].isnull(),'910_dst']
alpha.loc[alpha['910_dst'].isnull(),'910_dst']=alpha.loc[alpha['910_dst'].isnull(),'920_dst']
alpha.loc[alpha['920_dst'].isnull(),'920_dst']=alpha.loc[alpha['920_dst'].isnull(),'930_dst']
alpha.loc[alpha['930_dst'].isnull(),'930_dst']=alpha.loc[alpha['930_dst'].isnull(),'940_dst']
alpha.loc[alpha['940_dst'].isnull(),'940_dst']=alpha.loc[alpha['940_dst'].isnull(),'950_dst']
alpha.loc[alpha['950_dst'].isnull(),'950_dst']=alpha.loc[alpha['950_dst'].isnull(),'960_dst']
alpha.loc[alpha['960_dst'].isnull(),'960_dst']=alpha.loc[alpha['960_dst'].isnull(),'970_dst']
alpha.loc[alpha['970_dst'].isnull(),'970_dst']=alpha.loc[alpha['970_dst'].isnull(),'980_dst']
alpha.loc[alpha['980_dst'].isnull(),'980_dst']=alpha.loc[alpha['980_dst'].isnull(),'990_dst']


alpha.loc[alpha['800_dst'].isnull(),'800_dst']=alpha.loc[alpha['800_dst'].isnull(),'810_dst']
alpha.loc[alpha['810_dst'].isnull(),'810_dst']=alpha.loc[alpha['810_dst'].isnull(),'820_dst']
alpha.loc[alpha['820_dst'].isnull(),'820_dst']=alpha.loc[alpha['820_dst'].isnull(),'830_dst']
alpha.loc[alpha['830_dst'].isnull(),'830_dst']=alpha.loc[alpha['830_dst'].isnull(),'840_dst']
alpha.loc[alpha['840_dst'].isnull(),'840_dst']=alpha.loc[alpha['840_dst'].isnull(),'850_dst']
alpha.loc[alpha['850_dst'].isnull(),'850_dst']=alpha.loc[alpha['850_dst'].isnull(),'860_dst']
alpha.loc[alpha['860_dst'].isnull(),'860_dst']=alpha.loc[alpha['860_dst'].isnull(),'870_dst']
alpha.loc[alpha['870_dst'].isnull(),'870_dst']=alpha.loc[alpha['870_dst'].isnull(),'880_dst']
alpha.loc[alpha['880_dst'].isnull(),'880_dst']=alpha.loc[alpha['880_dst'].isnull(),'890_dst']
alpha.loc[alpha['890_dst'].isnull(),'890_dst']=alpha.loc[alpha['890_dst'].isnull(),'900_dst']

alpha.loc[alpha['710_dst'].isnull(),'710_dst']=alpha.loc[alpha['710_dst'].isnull(),'720_dst']
alpha.loc[alpha['720_dst'].isnull(),'720_dst']=alpha.loc[alpha['720_dst'].isnull(),'730_dst']
alpha.loc[alpha['730_dst'].isnull(),'730_dst']=alpha.loc[alpha['730_dst'].isnull(),'740_dst']
alpha.loc[alpha['740_dst'].isnull(),'740_dst']=alpha.loc[alpha['740_dst'].isnull(),'750_dst']
alpha.loc[alpha['750_dst'].isnull(),'750_dst']=alpha.loc[alpha['750_dst'].isnull(),'760_dst']
alpha.loc[alpha['760_dst'].isnull(),'760_dst']=alpha.loc[alpha['760_dst'].isnull(),'770_dst']
alpha.loc[alpha['770_dst'].isnull(),'770_dst']=alpha.loc[alpha['770_dst'].isnull(),'780_dst']
alpha.loc[alpha['780_dst'].isnull(),'780_dst']=alpha.loc[alpha['780_dst'].isnull(),'790_dst']
alpha.loc[alpha['790_dst'].isnull(),'790_dst']=alpha.loc[alpha['790_dst'].isnull(),'800_dst']

alpha.loc[alpha['700_dst'].isnull(),'700_dst']=alpha.loc[alpha['700_dst'].isnull(),'710_dst']
alpha.loc[alpha['690_dst'].isnull(),'690_dst']=alpha.loc[alpha['690_dst'].isnull(),'700_dst']
alpha.loc[alpha['680_dst'].isnull(),'680_dst']=alpha.loc[alpha['680_dst'].isnull(),'690_dst']
alpha.loc[alpha['670_dst'].isnull(),'670_dst']=alpha.loc[alpha['670_dst'].isnull(),'680_dst']
alpha.loc[alpha['660_dst'].isnull(),'660_dst']=alpha.loc[alpha['660_dst'].isnull(),'670_dst']
alpha.loc[alpha['650_dst'].isnull(),'650_dst']=alpha.loc[alpha['650_dst'].isnull(),'660_dst']

#======================================================================================================

beta.loc[beta['900_dst'].isnull(),'900_dst']=beta.loc[beta['900_dst'].isnull(),'910_dst']
beta.loc[beta['910_dst'].isnull(),'910_dst']=beta.loc[beta['910_dst'].isnull(),'920_dst']
beta.loc[beta['920_dst'].isnull(),'920_dst']=beta.loc[beta['920_dst'].isnull(),'930_dst']
beta.loc[beta['930_dst'].isnull(),'930_dst']=beta.loc[beta['930_dst'].isnull(),'940_dst']
beta.loc[beta['940_dst'].isnull(),'940_dst']=beta.loc[beta['940_dst'].isnull(),'950_dst']
beta.loc[beta['950_dst'].isnull(),'950_dst']=beta.loc[beta['950_dst'].isnull(),'960_dst']
beta.loc[beta['960_dst'].isnull(),'960_dst']=beta.loc[beta['960_dst'].isnull(),'970_dst']
beta.loc[beta['970_dst'].isnull(),'970_dst']=beta.loc[beta['970_dst'].isnull(),'980_dst']
beta.loc[beta['980_dst'].isnull(),'980_dst']=beta.loc[beta['980_dst'].isnull(),'990_dst']

beta.loc[beta['800_dst'].isnull(),'800_dst']=beta.loc[beta['800_dst'].isnull(),'810_dst']
beta.loc[beta['810_dst'].isnull(),'810_dst']=beta.loc[beta['810_dst'].isnull(),'820_dst']
beta.loc[beta['820_dst'].isnull(),'820_dst']=beta.loc[beta['820_dst'].isnull(),'830_dst']
beta.loc[beta['830_dst'].isnull(),'830_dst']=beta.loc[beta['830_dst'].isnull(),'840_dst']
beta.loc[beta['840_dst'].isnull(),'840_dst']=beta.loc[beta['840_dst'].isnull(),'850_dst']
beta.loc[beta['850_dst'].isnull(),'850_dst']=beta.loc[beta['850_dst'].isnull(),'860_dst']
beta.loc[beta['860_dst'].isnull(),'860_dst']=beta.loc[beta['860_dst'].isnull(),'870_dst']
beta.loc[beta['870_dst'].isnull(),'870_dst']=beta.loc[beta['870_dst'].isnull(),'880_dst']
beta.loc[beta['880_dst'].isnull(),'880_dst']=beta.loc[beta['880_dst'].isnull(),'890_dst']
beta.loc[beta['890_dst'].isnull(),'890_dst']=beta.loc[beta['890_dst'].isnull(),'900_dst']

beta.loc[beta['710_dst'].isnull(),'710_dst']=beta.loc[beta['710_dst'].isnull(),'720_dst']
beta.loc[beta['720_dst'].isnull(),'720_dst']=beta.loc[beta['720_dst'].isnull(),'730_dst']
beta.loc[beta['730_dst'].isnull(),'730_dst']=beta.loc[beta['730_dst'].isnull(),'740_dst']
beta.loc[beta['740_dst'].isnull(),'740_dst']=beta.loc[beta['740_dst'].isnull(),'750_dst']
beta.loc[beta['750_dst'].isnull(),'750_dst']=beta.loc[beta['750_dst'].isnull(),'760_dst']
beta.loc[beta['760_dst'].isnull(),'760_dst']=beta.loc[beta['760_dst'].isnull(),'770_dst']
beta.loc[beta['770_dst'].isnull(),'770_dst']=beta.loc[beta['770_dst'].isnull(),'780_dst']
beta.loc[beta['780_dst'].isnull(),'780_dst']=beta.loc[beta['780_dst'].isnull(),'790_dst']
beta.loc[beta['790_dst'].isnull(),'790_dst']=beta.loc[beta['790_dst'].isnull(),'800_dst']

beta.loc[beta['700_dst'].isnull(),'700_dst']=beta.loc[beta['700_dst'].isnull(),'710_dst']
beta.loc[beta['690_dst'].isnull(),'690_dst']=beta.loc[beta['690_dst'].isnull(),'700_dst']
beta.loc[beta['680_dst'].isnull(),'680_dst']=beta.loc[beta['680_dst'].isnull(),'690_dst']
beta.loc[beta['670_dst'].isnull(),'670_dst']=beta.loc[beta['670_dst'].isnull(),'680_dst']
beta.loc[beta['660_dst'].isnull(),'660_dst']=beta.loc[beta['660_dst'].isnull(),'670_dst']
beta.loc[beta['650_dst'].isnull(),'650_dst']=beta.loc[beta['650_dst'].isnull(),'660_dst']

Xtrain[dst_list] = np.array(alpha)
Xtest[dst_list] = np.array(beta)

for col in dst_list:  # 주파수는 거리 제곱에 비례 
    Xtrain[col] = Xtrain[col] * (Xtrain['rho'] ** 2)
    Xtest[col] = Xtest[col] * (Xtest['rho'] ** 2)
    
    

gap_feature_names=[]
for i in range(650, 1000, 10):
    gap_feature_names.append(str(i) + '_gap')

alpha=pd.DataFrame(np.array(Xtrain[src_list]) - np.array(Xtrain[dst_list]), columns=gap_feature_names, index=train.index)
beta=pd.DataFrame(np.array(Xtest[src_list]) - np.array(Xtest[dst_list]), columns=gap_feature_names, index=test.index)

Xtrain=pd.concat((Xtrain, alpha), axis=1)
Xtest=pd.concat((Xtest, beta), axis=1)

epsilon=1e-10

for dst_col, src_col in zip(dst_list, src_list):
    dst_val=Xtrain[dst_col]
    src_val=Xtrain[src_col] + epsilon
    delta_ratio = dst_val / src_val
    Xtrain[dst_col + '_' + src_col + '_ratio'] = delta_ratio
    
    dst_val=Xtest[dst_col]
    src_val=Xtest[src_col] + epsilon
    
    delta_ratio = dst_val / src_val
    Xtest[dst_col + '_' + src_col + '_ratio'] = delta_ratio
    
alpha_real=Xtrain[dst_list]
alpha_imag=Xtrain[dst_list]

beta_real=Xtest[dst_list]
beta_imag=Xtest[dst_list]

for i in tqdm(alpha_real.index):
    alpha_real.loc[i]=alpha_real.loc[i] - alpha_real.loc[i].mean()
    alpha_imag.loc[i]=alpha_imag.loc[i] - alpha_real.loc[i].mean()
    
    alpha_real.loc[i] = np.fft.fft(alpha_real.loc[i], norm='ortho').real
    alpha_imag.loc[i] = np.fft.fft(alpha_imag.loc[i], norm='ortho').imag

    
for i in tqdm(beta_real.index):
    beta_real.loc[i]=beta_real.loc[i] - beta_real.loc[i].mean()
    beta_imag.loc[i]=beta_imag.loc[i] - beta_imag.loc[i].mean()
    
    beta_real.loc[i] = np.fft.fft(beta_real.loc[i], norm='ortho').real
    beta_imag.loc[i] = np.fft.fft(beta_imag.loc[i], norm='ortho').imag
    
real_part=[]
imag_part=[]

for col in dst_list:
    real_part.append(col + '_fft_real')
    imag_part.append(col + '_fft_imag')
    
alpha_real.columns=real_part
alpha_imag.columns=imag_part
alpha = pd.concat((alpha_real, alpha_imag), axis=1)

beta_real.columns=real_part
beta_imag.columns=imag_part
beta=pd.concat((beta_real, beta_imag), axis=1)

Xtrain=pd.concat((Xtrain, alpha), axis=1)
Xtest=pd.concat((Xtest, beta), axis=1)


Xtrain=Xtrain.drop(columns=src_list)
Xtest=Xtest.drop(columns=src_list)

print(Xtrain.shape, Ytrain.shape, Xtest.shape)
model_scoring_cv(multi_model, Xtrain, Ytrain)


multi_model.fit(Xtrain, Ytrain)
preds=multi_model.predict(Xtest)


preds=pd.DataFrame(data=preds, columns=submission.columns, index=submission.index)
preds.head()
print(preds)

predict2=pd.read_csv('./data/dacon/comp1/predict.csv', index_col='id')
print("mae : ", mean_absolute_error(preds,predict2))

predict = pd.DataFrame(preds, columns=['hhb','hbo2','ca','na'])
predict.index = np.arange(10000,20000)
predict.to_csv('./data/dacon/comp1/predict2.csv', index_label='id')