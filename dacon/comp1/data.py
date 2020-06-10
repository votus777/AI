
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold, cross_validate

# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

print(train.shape)

train = train.drop('840_dst', axis=1)

print(train.shape)




# rho = train.sort_values(by =['rho'], axis=0)
# rho.to_csv('./data/dacon/comp1/rho.csv', index_label='id')

# sns.pairplot(train, diag_kind='hist')
# plt.show()

# corr_matrix = train.corr()
# hbb = corr_matrix["na"].sort_values(ascending=True)
# print(hbb)

# sns.pairplot(train_dst, diag_kind='hist')


# import missingno as msno
# msno.matrix(train_dst)  # dst가 난리다 
# plt.show()

# train.hist(bins=50, figsize=(20,20))


# plt.figure(figsize=(4, 12))
# sns.heatmap(train.corr().loc['rho':'990_dst', 'hhb':].abs())

# test.filter(regex='_dst$',axis=1).tail().T.plot() 
# plt.show()


# train = train.drop('840_dst', axis=1)
# test = test.drop(['810_dst','830_dst','840_dst'], axis=1 )