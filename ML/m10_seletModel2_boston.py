
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.utils.testing import all_estimators



boston = pd.read_csv('./data/csv/boston_house_prices.csv', header = 2, )

x = boston.iloc[ :, 0 : 12 ]
y = boston.iloc[:,13]


print(x.shape)
print(y.shape)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 31)



allAlgorithms = all_estimators(type_filter = 'regressor')


import warnings
warnings.filterwarnings('ignore')


for (name, algorithm) in allAlgorithms :
    model = algorithm()
    model.fit(x_train,y_train)
    

    score = model.score(x_test,y_test)
    print(name, "의 정답률 = ", score)

import sklearn
print(sklearn.__version__)

'''

ARDRegression 의 정답률 =  0.8048729456390089
AdaBoostRegressor 의 정답률 =  0.8686205202558096
BaggingRegressor 의 정답률 =  0.8835655852067814
BayesianRidge 의 정답률 =  0.8006873098781373
CCA 의 정답률 =  0.8042468729249165
DecisionTreeRegressor 의 정답률 =  0.7992949474313227
ElasticNet 의 정답률 =  0.6891452412884856
ElasticNetCV 의 정답률 =  0.6619327409990788
ExtraTreeRegressor 의 정답률 =  0.8433698615638954
ExtraTreesRegressor 의 정답률 =  0.9116938177212702
GaussianProcessRegressor 의 정답률 =  -5.9151057378494265
GradientBoostingRegressor 의 정답률 =  0.8668831874230866
HuberRegressor 의 정답률 =  0.8060347660514549
KNeighborsRegressor 의 정답률 =  0.5888650739150421
KernelRidge 의 정답률 =  0.8065580689114242
LarsCV 의 정답률 =  0.7758234608575244
Lasso 의 정답률 =  0.7589241986327055
LassoCV 의 정답률 =  0.778196163818867
LassoLars 의 정답률 =  -0.009165144612606202
LassoLarsCV 의 정답률 =  0.8110831082300849
LassoLarsIC 의 정답률 =  0.8081009362983141
LinearRegression 의 정답률 =  0.8081009362983163
LinearSVR 의 정답률 =  0.5909492075895797
MLPRegressor 의 정답률 =  0.41197342571863116


'''