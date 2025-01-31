
from sklearn.svm import LinearSVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score



# 1. 데이터 

x_data = [ [0,0], [1,0], [0,1], [1,1]] # -> and 연산 
y_data = [ 0, 0, 0, 1]

# 2. 모델 

model = LinearSVC()


# 3. 훈련 

model.fit(x_data, y_data)


# 4. 평가 및 예측

x_test = [ [0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0,0,0,1],y_predict)  # keras에서 evaulate

print(x_test, "의 예측 결과", y_predict)
print("acc : ", acc)


