

# 121p
c = [ "dog", "blue", "yellow"]

c = c + ["green"]  # 이떄는 append와 다르게 리스트의 결합이므올 [ ]로 감싸야한다
# c.append("green") 이라고 해도 된다 


# 131p
x = 5
while x !=0 :   # != 0  //   '0이 아닐 떄'
    x-= 1
    print(x)


# 133p 
animals = ["tiger", "dog", "elephant"]
for animal in animals :
    print(animal)

# 136p
storages = [ 1,2,3]
for n in storages :
    if n ==2 :
        continue
    print(n)

# continue -> 특정 조건일 때 루프를 한 번 건너 뛴다 

#137p
list = [ "a", "b"]
for index, value in enumerate(list):
    print(index, value)

# 리스트의 인덱스 확인 



