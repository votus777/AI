
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

x_train,x_test ,y_train, y_test = train_test_split(
    cancer.data, cancer.target, shuffle = True, train_size = 0.8, random_state = 31
)


model = DecisionTreeClassifier(max_depth=4)

model.fit(x_train,y_train)


acc = model.score(x_test,y_test)


print(model.feature_importances_)
'''
[0.         0.         0.         0.00967723 0.         0.
 0.0125804  0.         0.         0.         0.         0.
 0.         0.00516531 0.         0.         0.         0.
 0.         0.         0.         0.         0.80311759 0.02867123
 0.         0.00524183 0.01776586 0.11778056 0.         0.        ]

 '''