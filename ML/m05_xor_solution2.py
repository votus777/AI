
from sklearn.svm import LinearSVC , SVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from m03_xor.py

# 1. 데이터 

x_data = [ [0,0], [1,0], [0,1], [1,1]] # -> xor 연산 //  같으면 0, 다르면 1
y_data = [ 0, 1, 1, 0]

# 2. 모델 

# model = LinearSVC()
model = KNeighborsClassifier(n_neighbors=1)  

'''  
  최근접 이웃을 몇개씩 이을 것인지 설정 // 유유상종 

  무조건 1을 넣는다고 정확하다고 볼 수 없음 
  n = 1일 때 가장 가까운게 파란색이라면 결과값을 파란색으로 분류하겠지만 
  n = 4로 범위을 넓혀보면 더 큰 범위 안에서는 빨간색 공이 더 많이 있을 수 있기 때문이다.
  이때는 결과값을 빨간색으로 판단할 것이다. 즉, n 값의 변화에 따라 예측이 달라진다. 
  
  그러므로 최적의 n 값을 항상 찾아야한다. 
  (일반적으로 총데이터의 제곱근값을 사용한다)
  

  https://gomguard.tistory.com/51?category=712467

'''
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