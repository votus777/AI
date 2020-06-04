
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn  import datasets 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC , SVC 


# 데이터 
iris = datasets.load_iris()

x = iris.data
y = iris.target 


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8 )


minmax_scaler =MinMaxScaler()
x_train = minmax_scaler.fit_transform(x_train)
x_test = minmax_scaler.fit_transform(x_test)

# 모델 

model = SVC()      # score :  0.9333333333333333
# model = LinearSVC()   # score :  0.9333333333333333
# model =KNeighborsClassifier()   # score :  1.0
# model = RandomForestClassifier()  # score :  0.9333333333333333
                      
# model = RandomForestRegressor(n_estimators=1)  # acc :  0.8666666666666667                score :  0.8324022346368716
# model = KNeighborsRegressor(n_neighbors=1)  # 나오긴 나온다 acc :  0.8666666666666667    score (여기서는 R2) :  0.752577319587629           


# 훈련

model.fit(x_train, y_train)


# 평가 

y_predict = model.predict(x_test)

acc = accuracy_score(y_test,y_predict)  
score = model.score(x_test,y_test)

print("acc : ", acc)
print("score : ", score)

