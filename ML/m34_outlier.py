import numpy as np

def outliers(data): 
    quartile_1, quartile_3 = np.percentile(data, [25,75])
    print("1사분위 : ",quartile_1)
    print("3사분위 : ",quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr *1.5)
    upper_bound = quartile_3 + (iqr *1.5)
    
    return np.where((data > upper_bound) | (data < lower_bound))


a = np.array([1,2,3,4,10000,6,7,5000,90,100])


b = outliers(a)
print("이상치의 위치 : ", b)

'''

1사분위 :  3.25
3사분위 :  97.5
이상치의 위치 :  (array([4, 7], dtype=int64),)

'''