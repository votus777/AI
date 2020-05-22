# Example

# 실행이 되는 것들을 만들어라 

# ~ 120p 


from matplotlib import pyplot as plt  

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 3863.2, 5979.6, 10389.7, 14853.3]

# X  축에 연도,  y축에  GDP가 있는 선 그래프를 만들자 
plt.plot(years, gdp, color ='green', marker ='o', linestyle = 'solid')

# 제목을 더하자 
plt.title("Normal GDP")

#Y 축에 레이블을 추가하자 
plt.ylabel("Billions of $")
plt.show() 





