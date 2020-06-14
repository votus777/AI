import numpy as np
import matplotlib.pyplot as plt


# 299p
# 정규 분포에 따르는 난수 생성

np.random.seed(0) # 시드값 0으로 설정

x = np.random.randn(10000) # 정규분포르 따르는 난수 10000개 생성하여 변수 X에 대입 

nums = np.random.binomial(100, 0.5, size = 10000) # 성공확률 0.5로 100번 시도했을때 성공횟수를 구하는 실험 10000회 반속 

y = np.random.choice(x,5) # 리스트x에 있는 데이터 중 5개 무작위로 선택 

import datetime as dt

x = dt.datetime(1992,10,22)

print(x)  #1992-10-22 00:00:00

x = dt.timedelta(hours = 1, minutes= 30) # 시간의 길이를 나타내는 timedelta

y = x + dt.timedelta(1)  # 시간 연산 

s = "1992-10-22"
x = dt.datetime.strptime(s,  "Y-%m-%d" ) # 문자열로 datetime 객체 만들기 



# 308p

x = np.arange(0 ,11,2) # 0부터 10까지 짝수열 대입 
x = np.linespace(0,10,5) # 0부터 10까지 5개의 동일 간격으로 나누기 


