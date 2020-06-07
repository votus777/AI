

# 148p  매소드
number = [ 1, 5, 3, 4, 2]
print(sorted(number))  #[1, 2, 3, 4, 5]   # -> 기존 리스트 변경 X
print(number)   # [1, 5, 3, 4, 2]

number.sort()    #  -> 기존 리스트 변경  O
print(number)  # [1, 2, 3, 4, 5]



#149p
city = "Tokyo"
print(city.upper())   # TOKYO
print(city.count("o"))  # 2


#151p
fruit = "바나나"
color = "노란색"

print("{}는 {} 입니다.". format(fruit,color))  # 바나나는 노란색 입니다.

#152p
n = [ 3, 6, 8, 6, 3, 2, 4, 6]
print(n.index(2)) # 5
print(n.count(6)) # 3


#163p
import time  
now_time = time.time()
print(now_time)   # 1591531544.8309383

from time import time  # from 패키지명 import 모듈
now_time = time()
print(now_time)  # 1591531619.6661258

#


