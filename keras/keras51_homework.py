
#keras51_homework.py


import numpy as np

y = np.array([1,2,3,4,5,1,2,3,4,5])
y = np.array([0,1,2,3,4,0,1,2,3,4])


from keras.utils import np_utils

y = np_utils.to_categorical(y)
# y = y - 1



'''

print(y) 

[[0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1.]]

 

#_____________________ 1. slicing 하는 방법_______________________

y = y [ : , 1 : ]



#__________________ 2. y 전체에서 1을 빼는 방법 (numpy)______________


y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1

-> ([0,1,2,3,4,0,1,2,3,4])

numpy 안에서만 가능 ( 단, 같은 자료형만 가능 )

[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]




# ___________________3. sklearn 이용하는 방법___________________

#  그런데 y의 차원을 2차원으로 바꾸어 주어야 함 

print(y.shape) # (10,)

y = y.reshape(10,1) 


from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()


print(y)

[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]


 

_______________________________

Y_train = to_categorical(Y_train, num_classes = 10)  나중에 이것도 확인해보자 



np.argmax (a, axis) 


axis = 0  -> x 축 

axis = 1 -> y 축

axis = 2 -> z 축 


 '''