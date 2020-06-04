
import numpy as np

from keras import Sequential
from keras.layers import Dense, Softmax, Input
from keras.models import Model


# from m03_xor.py

# 1. 데이터 

x_data = [[0,0], [1,0], [0,1], [1,1]] # -> xor 연산 //  같으면 0, 다르면 1
y_data = [ 0, 1, 1, 0 ]

x_data = np.array(x_data)   # 기존 ML에서는 리스트 형태로 넣어도 가능하지만
y_data = np.array(y_data)   # Keras에서는 numpy 연산 때문에 배열로 변환 시켜주어야 한다. 

print(x_data.shape)  # (4,2)
print(y_data.shape)  # (4,)



# 2. 모델 

model = Sequential()   # 사실 이건 lin = LinearSVC(), sv = SVC() 이런 거랑 똑같음 
model.add(Dense(1, input_dim=2))
model.add(Dense(8))
model.add(Dense(8,activation='sigmoid'))
model.add(Dense(1))


# 3. 훈련 

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(x_data, y_data, epochs = 1000, batch_size= 1)


# 4. 평가 및 예측

x_test = [[0,0], [1,0], [0,1], [1,1]]
x_test = np.array(x_test)

print(x_test.shape)   # (4, 2)

y_test = np.array([0,1,1,0])

print(y_test.shape)  # (4,)
y_predict = model.predict(x_test)

print(y_predict.shape) # (4,1)

acc = model.evaluate(x_data,y_data)  

print(x_test, "의 예측 결과", y_predict)
print("loss, acc : ", acc)



'''
[[0 0]
 [1 0]
 [0 1]
 [1 1]] 의 예측 결과 
 
 [[-0.0061387 ]
 [ 0.9983316 ]
 [ 0.9966075 ]
 [ 0.16068012]]



loss, acc :  [0.045057766139507294, 1.0]

scikit-learn                       0.22.1

'''