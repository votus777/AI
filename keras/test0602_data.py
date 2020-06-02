
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import csv
# 데이터 불러오기 

samsung_datasets = pd.read_csv("./data/samsung.csv", index_col = 0, header = 0, sep = ',', encoding = 'ISO-8859-1',
                                                skiprows= 1, nrows=508, names=['일자','시가'])   #2020/06/01 ~  2018/05/04


hite_datasets = pd.read_csv("./data/hite.csv", index_col = 0, header = 0, sep = ',', encoding=  'ISO-8859-1', 
                                                skiprows=1, nrows=508, names=['일자','시가','고가','저가','종가','거래량' ])  # 2020/06/01 ~ 2018/05/04



print(samsung_datasets.shape)   #   (508, 1)
print(hite_datasets.shape)      #   (508, 5)    # 일자 - 시가 - 고가 - 저가 - 종가 - 거래량
 


for i in range(len(samsung_datasets.index)) :
    for j in range(len(samsung_datasets.iloc[i])):
        samsung_datasets.iloc[i,j] = int (samsung_datasets.iloc[i,j].replace(',',''))



for i in range(len(hite_datasets.index)) :
     for j in range(len(hite_datasets.iloc[i])):
        hite_datasets.iloc[i,j] = int (hite_datasets.iloc[i,j].replace(',',''))
  

samsung_datasets = samsung_datasets.sort_values(['일자'], ascending = [True])
hite_datasets = hite_datasets.sort_values(['일자'], ascending = [True])



samsung_datasets = samsung_datasets.values
hite_datasets = hite_datasets.values


print(type(samsung_datasets))  # <class 'numpy.ndarray'>
print(type(hite_datasets))   # <class 'numpy.ndarray'>



samsung_datasets = samsung_datasets.astype(int).astype(float)
hite_datasets = hite_datasets.astype(int).astype(float)



np.save('./data/samsung.npy',arr=samsung_datasets)
np.save('./data/hite.npy',arr=hite_datasets) 


'''
print(samsung_datasets)
[[53000.]   - 2018-05-04
 [52600.]   - 2018-05-08
 [52600.]
 [51700.]
 [52000.]
 [51000.]
 [50200.]
 [49200.]
 [50300.]
 [49900.]
 [49650.]
 [50600.]
 [52000.]
 [51000.]
 [52500.]
 [52200.]
 [51300.]
 [50400.]


print(hite_datasets)

 [  21400.   21600.   21350.   21550.  123592.]   - 2018-05-04
 [  21450.   21550.   21000.   21050.  250520.]   - 2018-05-08
 [  21050.   21200.   20950.   21100.  165195.]
 ...
 [  36200.   36300.   35500.   35800.  548493.]
 [  35900.   36750.   35900.   36000.  576566.]
 [  36000.   38750.   36000.   38750. 1407345.]]   -2020-06-01


 '''