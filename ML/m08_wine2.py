
# from m08_wine.py

import pandas as pd
import matplotlib.pyplot as plt
import sklearn

wine = pd.read_csv("./data/winequality-white.csv", sep = ';', header = 0)


y = wine['quality']
x = wine.drop('quality', axis = 1 )

print(x.shape)  # (4898, 11)
print(y.shape)  #(4898, )

# y 레이블 축소 

newlist = []
for i in list(y) :                 # 기존에 있던 3,4,5,6,7,8,9 를 0,1,2 세가지로 압축 
    if i <= 4:
        newlist += [0]
    elif i < 7 :
        newlist += [1]
    else : 
        newlist += [2]

y = newlist

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)



from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)

print("acc_score = ", accuracy_score(y_test,y_pred))
print("acc :  ", acc)


'''

acc_score =  0.8459183673469388
acc :   0.8459183673469388


'''