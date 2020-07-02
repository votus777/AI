

import numpy as np
import matplotlib.pyplot as plt 


# 수식
f = lambda x : x**2 -4*x + 6 

x = np.linspace(-1, 6, 100)
y = f(x)


# 그래프 그리기
plt.plot(x,y, 'k-')
plt.plot(2,2, 'sk')  # (2,2) 지점에 점을 찍는다 
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()