
import numpy as np
import matplotlib.pyplot as plt 

def sigmoid(x) :
    return 1 / ( 1 + np.exp(-x))  # np.exp = 밑(base)이 자연상수 e 인 지수함수로 변환
    
x = np.arange( -5, 5 , 0.1)
y = sigmoid(x)

print(x.shape, y.shape)

plt.plot(x,y)
plt.grid()
plt.show()

# activation -> 가중치 값을 한정시킨다. 

