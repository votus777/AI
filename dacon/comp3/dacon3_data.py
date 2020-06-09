
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# 데이터 

train = pd.read_csv('./data/dacon/comp3/train_features.csv', header = 0, index_col = 0)
target = pd.read_csv('./data/dacon/comp3/train_target.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', header = 0, index_col = 0)

# print(list(train.columns)) ['Time', 'S1', 'S2', 'S3', 'S4']

# print(train.shape)  # (1050001, 5)
# print(target.shape)  # (2800, 4)
# print(test.shape)   # (262500, 5)
 
# train - target   //   test - summit 

# train x : (2800, 375 ,4 )
# train target : (2800, 4)

# test x : ( 700, 375, 4)
# summit y : (700 ,4 )

# 와꾸만 맞아 보이는 3차원(lstm)은 Con1D로 풀어야 한다  -> Con는 특징 추출 

# 혹은 id당 모델을 돌려서 나온 x'값에 target 값과 비교하는 새로운 모델 생성 
# 그냥 x와 달리 x'는 시계열 데이터의 특징을 포함하고 있기 때문..? 

'''
print(train.head())


         Time   S1   S2   S3   S4
id
0.0  0.000000  0.0  0.0  0.0  0.0
0.0  0.000004  0.0  0.0  0.0  0.0
0.0  0.000008  0.0  0.0  0.0  0.0
0.0  0.000012  0.0  0.0  0.0  0.0
0.0  0.000016  0.0  0.0  0.0  0.0


print(train.info())

 #   Column  Non-Null Count    Dtype
---  ------  --------------    -----
 0   Time    1050001 non-null  float64
 1   S1      1050001 non-null  float64
 2   S2      1050000 non-null  float64
 3   S3      1049999 non-null  float64
 4   S4      1049999 non-null  float64



print(train.describe())

               Time            S1            S2            S3            S4
count  1.050001e+06  1.050001e+06  1.050000e+06  1.049999e+06  1.049999e+06
mean   7.479993e-04 -4.050979e+02 -4.050983e+02 -1.334345e+03 -1.605665e+03
std    4.330118e-04  2.753173e+05  2.753174e+05  2.655352e+05  3.026972e+05
min    0.000000e+00 -5.596468e+06 -5.596468e+06 -2.772952e+06 -6.069645e+06
25%    3.720000e-04 -7.426277e+04 -7.426321e+04 -7.855488e+04 -7.818371e+04
50%    7.480000e-04  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00
75%    1.124000e-03  7.391128e+04  7.391142e+04  7.295836e+04  7.665830e+04
max    1.496000e-03  3.865086e+06  3.865086e+06  3.655237e+06  3.687344e+06

'''

# train.hist(bins=50, figsize=(20,15))
# plt.show()


