import numpy as np
import matplotlib.pyplot as plt

# 다양한 그래프 그리기 

# 343p


days = np.arnge(1,11)
weight = ([10,14,16,17,13])

plt.plot(days, weight, marker ="o", markerfacecolor = "k", linestyle="--", color="b")

# 막대그래프 만들기

x = np.arange(1,11)
y1 = np.arange(11,21)
y2 = np.arange(5,10)

labels = ([1,2,3,4,5])

plt.bar(x,y1, tick_label = labels)
plt.bar(x,y2, bottom = y1)  # 누적막대그래프 


# 히스토그램 

plt.hist





