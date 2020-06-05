
from sklearn.svm import LinearSVC , SVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from sklearn import LDA

from sklearn.datasets  import load_boston


# 데이터
boston = load_boston()


x = boston.data
y = boston.target 


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(
   
    x, y, shuffle = True  , train_size = 0.8  
)



minmax_scaler =MinMaxScaler()
x_train = minmax_scaler.fit_transform(x_train)
x_test = minmax_scaler.fit_transform(x_test)

# standard_scaler = StandardScaler()
# x_train = standard_scaler.fit_transform(x_train)
# x_test = standard_scaler.fit_transform(x_test)



# 모델 

model = RandomForestRegressor()  # score :  0.8389419549293422
# model = KNeighborsRegressor()  # score :  0.8287281973662051

# model = SVC()      # error
# model = LinearSVC()   # error
# model =KNeighborsClassifier()   # error
# model = RandomForestClassifier()  # error
                      

# 훈련


model.fit(x_train, y_train)


# 평가 


y_predict = model.predict(x_test)

R2 = r2_score(y_test,y_predict)  
score = model.score(x_test,y_test)

print("r2 : ", R2)
print("score : ", score)      # R2와 똑같이 나옴 
