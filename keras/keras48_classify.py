
# keras48_classify.py


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = np.array(range(1, 11))
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  

'''

 1 or 0 -> Binary classification // 이진분류 

 activation - > sigmoid 함수 
 loss -> acc


'''


# x = np.reshape(x, (10,1,1))



# 2. 모델 구성


model = Sequential()
model.add(Dense(50, activation='relu', input_dim = 1))
model.add(Dense(20, activation= 'sigmoid'))
model.add(Dense(20, activation= 'sigmoid'))


model.add(Dense(1, activation= 'sigmoid'))

'''

ouput layer에 activation 함수 sigmoid를 추가해주어야 한다

'activation = ' => 활성함수 - 입력값이 특정 뉴런에서 처리 되어 결과값을 생성할 때 적용되는 함수

activation의 default 값은 ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나온다

'''
# 3. 컴파일, 훈련

from keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping( monitor='loss', patience= 50, mode ='auto')

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])

model.fit(x,y, epochs= 20000, batch_size= 2, callbacks= [early_stopping])

'''

이진분류에서는 loss = 'binary_crossentropy' 이거 쓴다. 

'''


# 평가 및 예측 




loss, acc = model.evaluate(x,y, batch_size=1)


x_predict = np.array([1,2,3])
y_predict = model.predict(x_predict)  

print('y_predict : ', np.around(y_predict))  
print('loss :', loss)
print('accuracy : ', acc)


'''
y_predict :  
[[0.50723827]
 [0.49775097]
 [0.48898444]]

loss : 0.6876743614673615
accuracy :  0.6000000238418579

값이 이상하게 나온다! 

예상 값은     1 , 0 , 1 이렇게 나와야 하는데 말이다. 



# 과제 y_predict 값이 0,1 이 나올 수 있도록 


// 하이퍼 파라미터 조정

이리저리 돌려보다가 loss 값이 계속 떨어진다 싶으면 epoch 왕창 늘려서 계속 내린다 

y_predict:

[[0.99871945] -> 1
 [0.0072667 ] -> 0
 [0.9887949 ]] -> 1

loss : 0.4804049411555752
accuracy :  0.699999988079071
______________________________________
y_predict :  
[[0.9998839]
 [0.0083991]
 [0.9700769]]
loss : 0.4792540667098365
accuracy :  0.699999988079071

역시 np.around(y_predict) 쓰니까

[[1.]
 [0.]
 [1.]]

깔끔하게 반올림해서 나온다 

'''