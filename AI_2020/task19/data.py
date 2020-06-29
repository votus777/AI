
import numpy as np
import pandas as pd


epm_train = pd.read_csv( 'AI_2020\\task19\\train_data\\train_EPM.csv', header = 0, index_col = 0)
swe_train = pd.read_csv('AI_2020\\task19\\train_data\\train_SWE.csv', index_col='time_tag', parse_dates=True)
xray_train = pd.read_csv('AI_2020\\task19\\train_data\\train_xray.csv',index_col='time_tag', parse_dates=True)
proton_train = pd.read_csv('AI_2020\\task19\\train_data\\train_proton.csv', header = 0, index_col = 0)

epm_val = pd.read_csv('AI_2020\\task19\\val_data\\val_EPM.csv', header = 0, index_col = 0)
swe_val = pd.read_csv('AI_2020\\task19\\val_data\\val_SWE.csv', index_col='time_tag', parse_dates=True)
xray_val = pd.read_csv('AI_2020\\task19\\val_data\\val_xray.csv', index_col='time_tag', parse_dates=True)
proton_val = pd.read_csv('AI_2020\\task19\\val_data\\val_proton.csv', header = 0, index_col = 0)

epm_test = pd.read_csv('AI_2020\\task19\\test_data\\test_EPM.csv', header = 0, index_col = 0)
swe_test = pd.read_csv('AI_2020\\task19\\test_data\\test_SWE.csv', index_col='time_tag', parse_dates=True)
xray_test = pd.read_csv('AI_2020\\task19\\test_data\\test_xray.csv', index_col='time_tag', parse_dates=True)
proton_test = pd.read_csv('AI_2020\\task19\\test_data\\test_proton.csv', header = 0, index_col = 0)
 

# print(epm_train.shape)       # (782974, 8)
# print(swe_train.shape)       # (3747609, 2)                               
# print(xray_train.shape)      # (3997440, 2)                               
# print(proton_train.shape)    # (799488, 1)                              
  
print(epm_val.shape)       # (782974, 8)
print(swe_val.shape)       # (3747609, 2)                               
print(xray_val.shape)      # (3997440, 2)                               
print(proton_val.shape)    # (799488, 1)      
                                 
# print(epm_test.shape)     #  (564540, 8)
# print(swe_test.shape)     # (2698655, 2)
# print(xray_test.shape)    # (2875680, 2)
# print(proton_test.shape)   # (575136, 1)





# 결측치 '-100'을 nan으로 바꾸기
epm_train = epm_train.replace(-100 , float("nan"))
swe_train = swe_train.replace(-100 , float("nan"))
xray_train = xray_train.replace(-100 , float("nan"))
proton_train = proton_train.replace(-100 , float("nan"))

epm_val = epm_val.replace(-100 , float("nan"))
swe_val = swe_val.replace(-100 , float("nan"))
xray_val = xray_val.replace(-100 , float("nan"))
proton_val = proton_val.replace(-100 , float("nan"))

epm_test = epm_test.replace(-100 , float("nan"))
swe_test = swe_test.replace(-100 , float("nan"))
xray_test = xray_test.replace(-100 , float("nan"))
proton_test = proton_test.replace(-100 , float("nan"))


# 결측치 보간 
epm_train = epm_train.interpolate(method = 'linear') 
swe_train = swe_train.interpolate(method = 'linear') 
xray_train = xray_train.interpolate(method = 'linear') 
proton_train = proton_train.interpolate(method = 'linear') 


epm_val = epm_val.interpolate(method = 'linear') 
swe_val = swe_val.interpolate(method = 'linear') 
xray_val = xray_val.interpolate(method = 'linear') 
proton_val = proton_val.interpolate(method = 'linear') 


epm_test = epm_test.interpolate(method = 'linear') 
swe_test = swe_test.interpolate(method = 'linear') 
xray_test = xray_test.interpolate(method = 'linear') 
proton_test = proton_test.interpolate(method = 'linear') 

epm_train = epm_train.fillna(0)
swe_train = swe_train.fillna(0)
xray_train = xray_train.fillna(0)
proton_train = proton_train.fillna(0)

epm_val = epm_val.fillna(0)
swe_val = swe_val.fillna(0)
xray_val = xray_val.fillna(0)
proton_val = proton_val.fillna(0)

epm_test = epm_test.fillna(0)
swe_test = swe_test.fillna(0)
xray_test = xray_test.fillna(0)
proton_test = proton_test.fillna(0)

# 다운 샘플링 
swe_train = swe_train.resample('5min').mean()
xray_train = xray_train.resample('5min').mean()

swe_val = swe_val.resample('5min').mean()
xray_val = xray_val.resample('5min').mean()


swe_test = swe_test.resample('5min').mean()
xray_test = xray_test.resample('5min').mean()


# Numpy array 변형
epm_train = epm_train.values
swe_train = swe_train.values
xray_train = xray_train.values
proton_train = proton_train.values

epm_val = epm_val.values
swe_val = swe_val.values
xray_val = xray_val.values
proton_val = proton_val.values

epm_test = epm_test.values
swe_test = swe_test.values
xray_test = xray_test.values
proton_test = proton_test.values



# 제로 패딩 
epm_train = np.pad(epm_train, ((16514, 0),(0,0)), 'constant', constant_values = 0)
epm_val = np.pad(epm_val, ((22594, 0),(0,0)), 'constant', constant_values = 0)
epm_test = np.pad(epm_test, ((11172, 0),(0,0)), 'constant', constant_values = 0)

epm_val = epm_val[ : -288, : ]
swe_val = swe_val[ : -288, : ]
xray_val = xray_val[ : -288, : ]


epm_test = epm_test[ : -576, : ]
swe_test = swe_test[ : -576, : ]
xray_test = xray_test[ : -576, : ]

print('=======================')   

print(epm_train.shape)       # (799488, 8)
print(swe_train.shape)       # (799488, 2)                             
print(xray_train.shape)      # (799488, 2)                              
print(proton_train.shape)    # (799488, 1)                              
   
print('=======================')   
                                 
print(epm_val.shape)         # (718560, 8)
print(swe_val.shape)         # (718560, 2)
print(xray_val.shape)        # (718560, 2)
print(proton_val.shape)      # (718560, 1)

print('=======================')   

print(epm_test.shape)        # (575136, 8)
print(swe_test.shape)        # (575136, 2)
print(xray_test.shape)       # (575136, 2)
print(proton_test.shape)     # (575136, 1)

# feature 병합 

x_train = np.concatenate((epm_train, swe_train, xray_train), axis = 1) 
x_val   = np.concatenate((epm_val, swe_val, xray_val), axis = 1) 
x_test  = np.concatenate((epm_test ,swe_test, xray_test), axis = 1) 


print(x_train.shape)         # (799488, 12)
print(x_val.shape)           # (718560, 12)
print(x_test.shape)          # (575136, 12)

print('x_train_front :', x_train[ :10])
print('x_val_front :', x_val[ : 10])
print('x_test_front : ', x_test[ :10])


print('x_train_back :', x_train[ -10 : ])
print('x_val_back :', x_val[ -10 : ])
print('x_test_back : ', x_test[ -10 : ])


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