
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


# 데이터 불러오기 

samsung_datasets = pd.read_csv("./data/samsung.csv", index_col = None, header = 0, sep = ',', encoding = 'ISO-8859-1')
 
kospi_datasets = pd.read_csv("./data/kospi200.csv", index_col = None, header = 0, sep = ',', encoding=  'ISO-8859-1')

'''
path = './data/samsung.csv'
with open(path, 'rb', encoding='utf-16') as f:
    content  = f.read()

path = './data/kospi.csv'
with open(path, 'rb', encoding='utf-16') as f:
    content  = f.read()
'''


print(samsung_datasets.shape)
print(kospi_datasets.shape)

for i in range(len(samsung_datasets.index)) :
    for j in range(len(samsung_datasets.iloc[i])):
        samsung_datasets.iloc[i,j] = int (samsung_datasets.iloc[i,j].replace(',',''))


for i in range(len(kospi_datasets.index)) :
    kospi_datasets.iloc[i,4] = int (kospi_datasets.iloc[i,4].replace(',',''))



samsung_datasets = samsung_datasets.sort_values(['일자'], ascending = [True])
kospi_datasets = kospi_datasets.sort_values(['일자'], ascending = [True])


samsung_datasets = samsung_datasets.values
kospi_datasets = kospi_datasets.values



