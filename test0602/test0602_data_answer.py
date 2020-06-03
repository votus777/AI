
import numpy as np
import pandas as pd



samsung = pd.read_csv("./data/samsung.csv", index_col = 0, header = 0, sep = ',', encoding = 'ISO-8859-1', names=['일자','시가'])


hite = pd.read_csv("./data/hite.csv", index_col = 0, header = 0, sep = ',', encoding = 'ISO-8859-1',names=['일자','시가','고가','저가','종가','거래량' ])  

print(samsung)
print(hite.head())
# print(samsung.shape)  # (700,1)
# print(hite.shape)     # (720,5)


#________________________ Non 제거 1 ________________________________

samsung = samsung.dropna(axis = 0)  # axis=0 -> x축 // axis = 1 -> y축

print(samsung)  
#print(samsung.shape)    # (509, 1)

# hite에도 dropna를 적용하면 6월 2일 데이터도 날라가버린다. 그래서 다른 방법 사용 
print(hite)
hite = hite.fillna(method = 'bfill')     # bfill -> backward fill 뒤에 것으로 채움 //  ffill -> foward fill 앞에 것으로 채움
hite = hite.dropna(axis =0)
print(hite)

#  samsung.fillna(smasung.mean())   // # filling missing values with mean per column


#________________________ Non 제거 2 __________________________________

# hite = hite [ 0 : 509]
# hite.iloc [0, 1:5] = [ 100, 200, 300, 400]   # 2020-06-02  39,000       10       20       30         40   -> 인덱스 위치 이용
# # hite.loc['2020-06-02', '고가': '거래량'] = [ '100', '200', '300', '400']                                 -> 헤더 이용 , str이기 떄문에 size+1  안해도 된다 

# print(hite)




# 삼성과 하이트의 정렬을 오름차순으로 변경 

samsung_datasets = samsung.sort_values(['일자'], ascending = [True])   # ascending - 오름차순  decending - 내림차순 
hite_datasets = hite.sort_values(['일자'], ascending = [True])  

# 콤마 제거, string을 int로 형변환 

for i in range(len(samsung.index)) :
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace( ',',''))   # 37,000 -> 37000

print(samsung)
print(type(samsung.iloc[0,0]))  # class 'int'

for i in range(len(hite.index)) :
    for j in range(len(hite.iloc[i])) :
        hite.iloc[i,j] = int (hite.iloc[i,j].replace(',','')) 


print(samsung.shape)  # (509,1)
print(hite.shape)    #  (509,5)



samsung = samsung.values
hite = hite.values


print(type(hite))  # class 'numpy_ndarray'


np.save('./data/samsung_answer.npy',arr=samsung)
np.save('./data/hite_answer.npy',arr=hite) 

print(samsung)
print(hite)