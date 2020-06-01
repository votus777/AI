import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv", index_col = None, header = 0, sep = ',')  # header = 0 -> 맨 윗줄은 데이터로 인식 X // sep -> , 로 데이터 구분을 하겠다 

print(datasets)

# 자나깨나 인덱스, 헤더 조심 

print(datasets.head())  # 위에서 다섯개

print(datasets.tail())  # 아래에서 다섯개 

#########################################
print("========== 중요 ===========")

print(datasets.values)   # pandas 를 numpy로 변환  

aaa = datasets.values

print(type(aaa))  # <class 'numpy.ndarray'>


print("===========================")
#########################################


np.save('./data/iris_aaa.npy', arr =aaa)



# numpy 저장 -> keras96.py  

