
import pandas as pd
from pyarrow import csv

# 데이터 

train = pd.read_csv('./data/dacon/comp3/train_features.csv', header = 0, index_col = 0)
target = pd.read_csv('./data/dacon/comp3/train_target.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', header = 0, index_col = 0)


from pycaret import regression

exp = regression.setup(data = train, target= 'S1')