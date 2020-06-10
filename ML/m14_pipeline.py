import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 데이터 
iris = load_iris()

x = iris.data
y = iris.target

# keras 는 x,y 한방에 떙기고 sklearn은 이렇게 나눠서 가져온다 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 42)



# 모델

# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([(("scaler"), MinMaxScaler()), (('svm'),SVC())])  # model 과 전처리 방법을 선택  
pipe = make_pipeline(MinMaxScaler(), SVC())

pipe.fit(x_train,y_train)

print("acc : " , pipe.score(x_test,y_test))

'''
acc :  1.0

'''