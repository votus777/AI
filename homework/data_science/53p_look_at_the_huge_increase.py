from collections import Counter
from matplotlib import pyplot as plt  


mention = [500, 505]
years = [2017, 2018]

plt.bar(years, mention, 0.8)
plt.xticks(years)
plt.ylabel("# of times i heard someone say date science")

 # 이렇게 하지 않으면 matplpotlib가 x축에 0,1 레이블을 달고
 # 주변부 어딘가에 +2.013e3 이라고  표기해 둘 것이다
# plt.ticklabel_format (useOffset = False) # 왜인지 오류나서 뺌
'''
 plt.axis([2016.5, 2018.5,  299, 506])
 plt. title("Look at the guge increase!")
 plt.show() 

'''