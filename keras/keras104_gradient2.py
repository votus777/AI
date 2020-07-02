

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
# plt.show()

gradient = lambda x : 2*x - 4 


x0 = 0.0 
MaxIter = 20 
learing_rate = 0.25

print("step\tx\tf(x)") # \t 띄어쓰기 
print("{:02d}\t{:6.5}\t{:6.5}".format( 0, x0, f(x0)))  # {6.5} 앞 6자리까지, 뒤 5자리 까지 

# f = lambda x : x**2 -4*x + 6 

# 가장 기본적인 경사하강법 파이썬 코드 
for i in range(MaxIter) : 
    x1 = x0 - learing_rate * gradient(x0)
    x0 = x1
    
    print("{:02d}\t{:6.5}\t{:6.5}".format( i+1, x0, f(x0)))
    
'''
step    x       f(x)
00         0.0     6.0
01         1.0     3.0
02         1.5    2.25
03        1.75  2.0625
04       1.875  2.0156
05      1.9375  2.0039
06      1.9688   2.001
07      1.9844  2.0002
08      1.9922  2.0001
09      1.9961     2.0
10       1.998     2.0
11       1.999     2.0
12      1.9995     2.0
13      1.9998     2.0
14      1.9999     2.0
15      1.9999     2.0
16         2.0     2.0
17         2.0     2.0
18         2.0     2.0
19         2.0     2.0
20         2.0     2.0

'''