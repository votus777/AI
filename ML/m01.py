
import numpy as np 
import matplotlib.pyplot as plt

x = np.array(0, 10, 0.1)
y = np.sin(x)

plt.plot (x,y)  # 0부터 10까지 0.1씩 증가하면서 각 0.1에 해당하는 sin값 출력 

plt.show()