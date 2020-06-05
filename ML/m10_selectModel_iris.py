
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.utils.testing import all_estimators



iris = pd.read_csv('./data/csv/iris.csv', header = 0, )

x = iris.iloc[ :, 0 : 4 ]

y = iris.iloc[:,4]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 31)



allAlgorithms = all_estimators(type_filter = 'classifier')


import warnings
warnings.filterwarnings('ignore')


for (name, algorithm) in allAlgorithms :
    model = algorithm()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 = ", accuracy_score(y_test,y_pred))

import sklearn
print(sklearn.__version__)


'''

AdaBoostClassifier 의 정답률 =  0.9
BaggingClassifier 의 정답률 =  0.9666666666666667
BernoulliNB 의 정답률 =  0.26666666666666666
CalibratedClassifierCV 의 정답률 =  0.8
ComplementNB 의 정답률 =  0.6
DecisionTreeClassifier 의 정답률 =  0.9333333333333333
ExtraTreeClassifier 의 정답률 =  0.8666666666666667
ExtraTreesClassifier 의 정답률 =  0.9
GaussianNB 의 정답률 =  0.9666666666666667
GaussianProcessClassifier 의 정답률 =  0.9666666666666667
GradientBoostingClassifier 의 정답률 =  0.9333333333333333
KNeighborsClassifier 의 정답률 =  0.9333333333333333
LabelPropagation 의 정답률 =  0.9333333333333333
LabelSpreading 의 정답률 =  0.9333333333333333
LinearDiscriminantAnalysis 의 정답률 =  0.9666666666666667
LinearSVC 의 정답률 =  0.8666666666666667
LogisticRegression 의 정답률 =  0.8333333333333334            ->   이름은 regression 인데 classifier이다 
LogisticRegressionCV 의 정답률 =  0.9333333333333333
MLPClassifier 의 정답률 =  0.9333333333333333
MultinomialNB 의 정답률 =  0.8
NearestCentroid 의 정답률 =  0.9333333333333333
NuSVC 의 정답률 =  0.9666666666666667
PassiveAggressiveClassifier 의 정답률 =  0.8666666666666667
Perceptron 의 정답률 =  0.7333333333333333
QuadraticDiscriminantAnalysis 의 정답률 =  0.9666666666666667
RadiusNeighborsClassifier 의 정답률 =  0.9666666666666667
RandomForestClassifier 의 정답률 =  0.9666666666666667
RidgeClassifier 의 정답률 =  0.6666666666666666
RidgeClassifierCV 의 정답률 =  0.6666666666666666
SGDClassifier 의 정답률 =  0.8666666666666667
SVC 의 정답률 =  0.9666666666666667
0.20.1

'''