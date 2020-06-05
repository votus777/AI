
# from m01_selectModel.py

import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.utils.testing import all_estimators

import warnings
warnings.filterwarnings('ignore')


# 1. 데이터 
iris = pd.read_csv('./data/csv/iris.csv', header = 0, )

x = iris.iloc[ :, 0 : 4 ]

y = iris.iloc[:,4]


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 31)

# K fold 사용
from sklearn.model_selection import KFold, cross_val_score


#############################################################################

kfold = KFold(n_splits=5, shuffle=True)  # 5개로 나누고,  검증이 5번 일어난다 


'''

k - fold 

K개의 fold를 만들어서 진행하는 교차검증 

-> 모델의 성능 평가를 일반화

1. 데이터 뻥튀기
2. Train // Validation // Test 3개로 나누는 것보다 Train // Test 두 개로 나누는 것이 샘플 당 더 많은 데이터가 들어갈 수 있음 -> overfitting 방지


노이즈 값이 큰 데이터들이 한 쪽에 쏠려서 제대로 검증 및 훈련이 이루어질 수 없을 때 유용하다

(핸즈온 머신러닝 교재 112p)


'''

#############################################################################

allAlgorithms = all_estimators(type_filter = 'classifier')



for (name, algorithm) in allAlgorithms :
    model = algorithm()
    
    # model.fit(x,y)
    # fit 하고 score가 섞인 녀석
    scores = cross_val_score(model,x,y, cv=kfold) # cv (cross-validation을 뭐로 할것이냐) = kfold 


    print(name, "의 정답률은 : ", scores)  # 여기서 score 는 acc

    # y_pred = model.predict(x)
    # print(name, "의 정답률 = ", accuracy_score(y,y_pred))

import sklearn
print(sklearn.__version__)

'''


'''