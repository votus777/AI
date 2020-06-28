import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

from statsmodels.tsa.arima_model import ARIMA

from keras.models import Sequential 
from keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten 

from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor, LGBMClassifier 


epm = pd.read_csv( 'AI_2020\\task19\\train_data\\train_EPM.csv', header = 0, index_col = 0)
swe = pd.read_csv('AI_2020\\task19\\train_data\\train_SWE.csv', header = 0, index_col = 0)
xray = pd.read_csv('AI_2020\\task19\\train_data\\train_xray.csv', header = 0, index_col = 0)
proton = pd.read_csv('AI_2020\\task19\\train_data\\train_proton.csv', header = 0, index_col = 0)

print(swe.iloc[10000])

print(proton.iloc[10000])




'''


- Input Data(X-ray, SWE, EPM) 중 -100은 위성의 고장으로 인해 해당 데이터가 관측이 되지 않았다는 메세지임. 
  (Baseline에서는 임의로 0으로 변환합니다. 해당 Empty Cell의 데이터를 추정, 예측하여 Output에 반영하는 것 또한 참가자의 Skill로 합니다.)



1998-02-04 00:05:00.000 to 2005-09-10 23:55:00.000
=========================================================
epm.info()    

 지구에서 태양방향으로 150만 km 떨어진 위치의 ACE 위성에서 관측된, 코로나물질방출에 의해 생성되는 고에너지 양성자와 전자 밀도의 5분 평균값  -> 지구도달 1~3일

 양성자의 Traval Time은 태양 플레어, 코로나물질방출의 발생 위치, 방향 등에 따라 유동적임

(782974, 8)    78만개      5분 평균값

10 50 100

Data columns (total 8 columns):

 #   Column                                    Non-Null Count   Dtype            print(epm.iloc[1])                              
---  ------                                    --------------   -----
 0   P1P_.047-.066MEV_IONS_1/(cm**2-s-sr-MeV)  782974 non-null  float64              173.340000
 1   P2P_.066-.114MEV_IONS_1/(cm**2-s-sr-MeV)  782974 non-null  float64              41.027000
 2   P3P_.114-.190MEV_IONS_1/(cm**2-s-sr-MeV)  782974 non-null  float64              20.590000
 3   P4P_.190-.310MEV_IONS_1/(cm**2-s-sr-MeV)  782974 non-null  float64              8.544700
 4   P5P_.310-.580MEV_IONS_1/(cm**2-s-sr-MeV)  782974 non-null  float64              1.815500
 5   P6P_.580-1.05MEV_IONS_1/(cm**2-s-sr-MeV)  782974 non-null  float64              0.713280
 6   P7P_1.05-1.89MEV_IONS_1/(cm**2-s-sr-MeV)  782974 non-null  float64              0.109980      
 7   P8P_1.89-4.75MEV_IONS_1/(cm**2-s-sr-MeV)  782974 non-null  float64              0.051257
 
dtypes: float64(8)
memory usage: 53.8+ MB

=======================================================
swe.info()  

(3747609, 2)  370만개    1분 평균값

지구에서 태양방향으로 150만 km 떨어진 위치의 ACE 위성에서 관측된, 태양에서 불어오는 태양풍의 밀도와 속도의 1분 평균값

Data columns (total 2 columns):
 #   Column           Dtype
---  ------           -----
 0   H_DENSITY_#/cc   float64            H_DENSITY_#/cc       5.7107   cm3
 1   SW_H_SPEED_km/s  float64            SW_H_SPEED_km/s    441.6400
 
dtypes: float64(2)
memory usage: 85.8+ MB

==================================================

xray.info()

(3997440, 2)  400만개   1분 평균값

지구 정지궤도에 있는 미국 GOES 위성에서 관측된 x-ray의 1분 평균값 -> 지구 도달 8분

Data columns (total 2 columns):
 #   Column  Dtype                   xray.iloc[1]
---  ------  -----
 0   xs      float64                 8.950000e-10  1.0 Angstrom에서 8.0 Angstrom
 1   xl      float64                 4.730000e-08  0.5 Angstrom에서 4.0 Angstrom
 
dtypes: float64(2)
memory usage: 91.5+ MB

==========================================================
proton.info()

(799488, 1)  80만개   5분 평균값 

Proton : 지구 정지궤도에 있는 미국 GOES 위성에서 관측된 10MeV 이상 Proton flux의 5분 평균값

- Output Data(Proton) 중 -100은 위성의 고장으로 인해 해당 데이터가 관측이 되지 않았다는 메시지입니다. 
  (Baseline에서 해당 Step은 Train에서 제외됩니다. 해당 Empty Cell의 데이터를 추정, 예측하여 Output에 반영하는 것 또한 참가자의 Skill로 합니다.)


Data columns (total 1 columns):
 #   Column  Non-Null Count   Dtype
---  ------  --------------   -----
 0   proton  799488 non-null  float64          proton    0.17
dtypes: float64(1) 
memory usage: 12.2+ MB 

===============================================================

'''