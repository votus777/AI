
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from xgboost  import XGBClassifier

cancer = load_breast_cancer()

x_train,x_test ,y_train, y_test = train_test_split(
    cancer.data, cancer.target, shuffle = True, train_size = 0.8, random_state = 31
)


# model = DecisionTreeClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth = 4)

model.fit(x_train,y_train)


acc = model.score(x_test,y_test)


print(model.feature_importances_)
print(acc)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importance_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


plot_feature_importance_cancer(model)
plt.show()

'''
[8.21806025e-03 1.49108805e-02 8.32287245e-04 3.38273086e-02
 6.38609985e-03 5.68289496e-03 0.00000000e+00 6.91523179e-02
 1.40147354e-03 1.16265284e-04 1.67121217e-02 6.59667654e-03
 4.89120081e-04 1.15110455e-02 3.23544629e-03 3.64983338e-03
 1.01513118e-02 1.05642644e-03 5.77634107e-03 7.80429645e-03
 1.70832098e-01 2.00546645e-02 4.34721768e-01 2.32134964e-02
 4.83889878e-03 6.49427343e-03 2.67045684e-02 1.01384960e-01
 4.24511172e-03 0.00000000e+00]
0.9736842105263158
'''