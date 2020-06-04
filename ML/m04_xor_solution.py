
from sklearn.svm import LinearSVC , SVC
from sklearn.metrics import accuracy_score

# from m03_xor.py

# 1. 데이터 

x_data = [ [0,0], [1,0], [0,1], [1,1]] # -> xor 연산 //  같으면 0, 다르면 1
y_data = [ 0, 1, 1, 0]

# 2. 모델 

# model = LinearSVC()
model = SVC()

# 3. 훈련 

model.fit(x_data, y_data)


# 4. 평가 및 예측

x_test = [ [0,0], [1,0], [0,1], [1,1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0,1,1,0],y_predict)  # keras에서 evaulate

print(x_test, "의 예측 결과", y_predict)
print("acc : ", acc)

'''

[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과 [0 1 1 0]
acc :  1.0

'''