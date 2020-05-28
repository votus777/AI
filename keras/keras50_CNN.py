
#keras50_CNN.py


from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D, Flatten, Dense   # import 해주자

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape = (10, 10, 1)))   #output ->  (9,9,10)            


model.add(Conv2D(7, (3,3)))                                   #output -> (7, 7, 7)
model.add(Conv2D(5, (2,2)))                                   #output -> (7, 7, 5)
model.add(Conv2D(5, (2,2)))                                   #output -> (6, 6, 5)

# model.add(Conv2D(5, (2,2),strides = 2))                     #output -> (3, 3, 5)

# model.add(Conv2D(5, (2,2), strides = 2, padding = 'same'))  #output -> (3, 3, 5)   // stride가 우선순위 

model.add(MaxPooling2D(pool_size= 2))                         #output -> (3, 3, 5)   // 여전히 4차원 텐서 output   ( none, 3, 3, 5)


model.add(Flatten())                                          

'''

  #  Dense 모델에 넣기 위해 4차원 텐서의 차원을 쫙 펴서 낯추자 -> Flattten 
  #  CNN의 끝은 항상 Flatten layer
  #  MaxPooling은 취향껏
   
'''                                                                
model.add(Dense(1)) 

model.summary()
'''
Model: "sequential_1"  
                  ____________padding = vaild_______________

_________________________________________________________________
Layer (type)                 Output Shape              Param #
                        //  input_shape (5, 5, 1)
=================================================================
conv2d_1 (Conv2D)            (None, 4, 4, 10)           50

                                  ->  10장의 이미지가 생겼다! 
                                  ->  입력 데이터는 채널 수와 상관없이 필터 별로 1개의 피처 맵이 만들어진다. 
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 10)          820
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 2, 2, 10)           405  ->  필터의 수 * ( 커널 사이즈 * 채널 수 + 1)                   output * (input * kernel * kernel+ bias)
                       
                        output shape가 점점 작아지고 있음 
                        
                    그런데 커널 사이즈로 잘랐을 떄 중첩되는 부분이 가운데에 집중되는 문제점이 있다. -> 중첩된 특성만 증폭, 사이드 쪽의 상대적인 데이터 손실 
                    그래서 padding을 입혀서 따뜻하게 해주자 -> 가변쪽에 0 더미 데이터를 만들어줌 
                    dafualt 값은 vaild -> (사이즈가 맞지 않을 경우 가장 우측의 열 혹은 가장 아래의 행을 드랍한다).

■ ■ ■ ■ ■
■ ■ ■ ■ ■         ■ ■ ■          ■ ■ ■
■ ■ ■ ■ ■     %   ■ ■ ■    ->    ■ ■ ■  
■ ■ ■ ■ ■         ■ ■ ■          ■ ■ ■
■ ■ ■ ■ ■

  5 * 5           3 * 3          3 * 3
       (stride = 1) 


Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 7)           637
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 6, 5)           145
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 5, 5)           105
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 2, 2, 5)           0
_________________________________________________________________
flatten_1 (Flatten)          (None, 20)                0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 21
=================================================================
Total params: 958
Trainable params: 958
Non-trainable params: 0




                       input :  4차원 텐서 ->  output : 4차원 텐서 
                            





'''
'''

Conv2D 레이어에 관하여 

(Conv1D, Conv3D 등등도 있는데 당장은 안쓸거다) 


model.add(Conv2D(10, (2,2), padding = 'same', input_shape = ( 5, 5, 1))) 
                 |     |        |                             행 열 채널수     
                 |     |        |                       'Batch_size'+(height , width, channels)
             필터의 수  |   출력크기를 조절-> 커널 사이즈에 관계없이 동일한 크기로 만들어줌                          
              (Depth)       |
                  커널 사이즈
                = kernel_size=2
                  
※ channels = input_dim           

※ stride -> 커널이 한 번에 움직이는 간격 (default = 1)
         1일 떄가 가장 효과적이지만 연산량을 줄이기 위해 1이 아닌 값을 적용하기도 한다. 

※ Maxpooling ->  pool_size ( n, n ) , 특성 중에 가장 중요한 것들만 추출한다 
                  이 레이어는 영상의 작은 변화라던지 사소한 움직임이 특징을 추출할 때 크게 영향을 미치지 않도록  
                  학습 속도 증가 및 overfitting 방지 



'''











