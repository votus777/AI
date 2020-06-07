import numpy as np

# 189p

storages = [ 1,2,3,4]
new_stroges = []
for n in storages :
    n += n
    new_stroges.append(n)
print(new_stroges)

stroges = np.array([1,2,3,4])
stroges += stroges
print(stroges)


#193p
# np.array의 슬라이스는 배열의 복사본이 아닌 view라는 점이다. 원래 ndarray를 변경하게 된다는 뜻
arr1 = np.array([1,2,3,4,5])

arr2 = arr1.copy()
#arr2 변수를 변경하면 원래 변수 (arr1)에 영향을 주지 않는다
arr2[0] = 100

print(arr1)  # [1 2 3 4 5]


#196p
#[] 안에 논리값 T/F 사용
arr = np.array([2,4,6,7])
print(arr[arr % 3 == 1])  # [4,7]

#197p 범용 함수 - 요소별로 계산

# np.abs()       -> 절대값 반환
# np.substract() -> 요소 간의 차이를 반환
# np.maximum()   -> 요소 간의 최대값 반환
# np.exp()       -> 요소의 e 의 거듭제곱을 반환
# np.sqrt()      ->  요소의 제곱근 반환



#199p 집합 함수  -> 1차원 배열만을 대상

# np.unique() -> 배열 요소에서 중복을 제거하고 정렬한 결과 반환
# np.union1d(x,y) -> 배열 x 와 y의 합집합을 정렬해서 반환
# np.intersect1d(x,y) -> 배열 x 와 y의 교집합을 정렬해서 반환
# np.setfiffd(x,y) -> 배열 x에서 y를 뺀 차집합을 정렬해서 반환





