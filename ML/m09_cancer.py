

from sklearn.svm import LinearSVC , SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from lightgbm import plot_importance, LGBMRegressor,LGBMClassifier

# 데이터 
cancer = load_breast_cancer()  #이진분류

x = cancer.data
y = cancer.target

print(x.shape)  #(569, 30)
print(y.shape)  #(569,)



from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)


print(x_train.shape)  # (455, 30)
print(x_test.shape)   # (114, 30)

print(y_train.shape)  # (455, )
print(y_test.shape)   # (114,)

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.fit_transform(x_test)

pca = PCA(n_components=15)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)


# 모델 

# model = SVC()       # score : 0.9210526315789473
# model = LinearSVC()  # score : 0.7807017543859649
# model =KNeighborsClassifier()   # score : 0.8421052631578947
# model = RandomForestClassifier()  # score : 0.8771929824561403
model = LGBMClassifier(n_estimators=800) # acc :  0.9473684210526315

# model = RandomForestRegressor()  # error
# model = KNeighborsRegressor()  # error               


# 훈련

model.fit(x_train, y_train)


# 평가 


y_predict = model.predict(x_test)

acc = accuracy_score(y_test,y_predict)  
score = model.score(x_test,y_test)

print("acc : ", acc)
print("score :",score)  # acc와 똑같이 나온다 
