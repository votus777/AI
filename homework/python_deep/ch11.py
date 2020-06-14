import numpy as np
import matplotlib.pyplot as plt

# 313p

x = np.linspace(0,2*np.pi)
y = np.sin(x)


plt.show(x,y)
# 그래프 그리기 



plt.ylim([0,1]) # y축의 표시범위 [0,1]

plt.title("y=sin(x)")
plt.xlabel("x축")
plt.ylabel("y축")


plt.gird(True) # 그래프에 그리드 표시하기 

# plt.xticks(positions, labels)  # (눈금 삽입 위치, 삽입할 눈금)

plt.plot(x,y, color = "b") # 그래프 색상 설정 
# "b" , "g", "r", "c"(청록색), "m"(진홍색), "y", "k"(검은색), "w"

plt.figure(figsize=(4,4)) # 그림 크기 설정 , inch 

