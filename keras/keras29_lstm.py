from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM


#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])  
y = array([4, 5, 6, 7]) 



print("x.shape : ", x.shape)    # (4,3)
print("y1.shape : ", y.shape)  # (4, )    


'''

여기서 (4,3)을 (4,3,1)로 바꿔주면 개체들이 각각 묶이게 된다!
-> [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]]]
-> 이제 한 개씩 작업할 수 있겠다 

'''
x = x.reshape(4, 3, 1)
# x = x.reshape(x.shape[0], x.shape[1], 1)

print(x.shape)  # (4, 3, 1)    # [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]]]

# 모두 곱해봐서 값이 맞는지 확인 


'''
y.shape  ->   왜 (4,1)가 아나고 (4,)일까?? 

(4,1) 해버리면 에러난다 난리난다 

(4,1)는 벡터이고 (4,)는 스칼라이므로 둘은 같지 않다(?)


(1) - array []
([1,2]) - array [2]
([1,2,3]) - array [3]
([[1,2,3], [4,5,6]]) -array(2,3)
([[1,2,3],[4,5,6],[7,8,9]]) - arrray(3,3)


1 - [[1,2,3], [1,2,3]] - array(2,3)
2 - [[[1,2],[4,3]], [[4,5],[5,6]]]- array(2,2,2)
3 - [[[1],[2],[3]],[[4],[5],[6]]] - array(2,3,1)
4 - [[[1,2,3,4]]] - array(1,1,4) 
5 - [[[[1],[2]]]],[[[3],[4]]]] - array (1,2,2,1)

# 헷갈리면 가장 작은 차원부터 올라오자  

'''

# 2. 모델 구성

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape = (3,1)))  

'''

결국 이것도 행무시와 같은 맥락 

[[a],[b],[c]] 형식의 데이터를 입력하겠다 선언

음 아마... 각각의 feature들을 구분해서 학습시키기 위해서(?)

Ex) 날씨 : 온도, 습도, 계절 등등 여러 요인들 
       -> 1월 1일의 날씨 데이터  [온도1, 습도1, 계절1], 2일의 데이터 [온도2, 습도2, 계절2] ...
       -> reshape을 안하면 [온도1, 습도1, 계절1]가 통째로 들어가게 되는데 
       -> [[온도],[습도],[계절]] 이렇게 바꾸면  요소들이 모델에 '각각' 들어갈 수 있게된다.
       -> 더욱 정확한 예측 가능

x에는 스칼라 3종류를 가진 2차원 데이터 '4개'가 있다. 다음에도 추가될 데이터도 같은 규격. 각각의 2차원 데이터를 input 하는 것

즉, 만약 1년치 데이터가 있을때, 그냥 365일 통째로 넣어도 되고 대략 (12,30) 으로 나눠서 넣어도 된다는 소리다.



'''
model.add(Dense(5))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(5))
model.add(Dense(1))

'''
model.summary()


Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 10)                480
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 541
Trainable params: 541
Non-trainable params: 0

레이어하고 노드를 얼마 넣지도 않았는데 params이 이 모양이다. 괜히 느리다고 하는게 아니다. 

*과제*  
왜 LSTM의 파라미터가 이렇게 높게 나오는 것일까? 


'''

# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, callbacks= [ealry_stopping])




x_input  = array([5,6,7])  # 평가를 위해 [5,6,7] 대입, 현재 형태는(1,3)
# print("x_input shape : ",x_input)  # [5,6,7]

x_input = x_input.reshape(1,3,1) #  x_input를 (1,3,1)로 reshape  
# print("x_input shape : ",x_input)
# x_input shape :  [[[5], [6], [7]]]


yhat = model.predict(x_input)
print(yhat)         #[[8.012411]] loss: 2.4653e-10 #나름 잘나왔다 
print(yhat.shape)   # (1,1)


'''
그러나 LSTM을 쓰기에는 너무 작은 데이터 


'''







