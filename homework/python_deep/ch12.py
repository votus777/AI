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

data = np.random.randn(1000)

plt.hist(data, bins=100, normed=True, cumulative =True) # 구간수 100, 정규화하기 , 누적화하기 

# 산포도 만들기
y = np.arange(11,21)
z = np.arrange(11,20)


plt.scatter(x,y, marker = "s", color = "k")  # marker -> "o" "s" " p" "*" "+" "D"


plt.scatter(x,y, s =z)   #Z 값에 따라서 마커 크기 변화
plt.scatter(x,y, c =z, cmap="Blues")   #Z 값에 따라서 마커 농도 변화 


plt.colorbar() # 컬러바 표시 

#원 그래프 만들기
# plt.pie(data, labels=labels, explode= explode)  # explode =>  해당 요소 부각 explode = [ 0,0,0.10,0]

plt.axis("equal") # 원형



#3D 그래프 만들기




