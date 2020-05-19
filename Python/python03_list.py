

#자료형 
# 1. 리스트 

a = [ 1,2,3,4,5]
b = [1,2,3,'a', 'b']
print(b)  #[1, 2, 3, 'a', 'b']

#list 안에 들어가는 자료를은 형이 달라도 되지만 
#numpy에 들어가는 데이터는 딱 한가지 자료형만 써야한다 

print(a[0] + a[3]) # 5 (1+4)
# print(b[0] + a[3]) # error   


print (type(a)) # list 

print (a[-2]) # 4
print (a[1:3]) #[2,3]

a = [1,2,3, ['a', 'b', 'c']]
print(a[1]) # 2
print(a[-1]) # ['a','b','c']
 
print (a[-1][1]) # b

# 1-2. 리스트 슬라이싱 

a = [1,2,3,4,5] 
print (a[:2]) # [1,2] 


# 1-3. 리스트 더하기 

a = [1,2,3]
b = [4,5,6]

print(a+b) #[1,2,3,4,5,6]? [5,7,9]?  -> [1, 2, 3, 4, 5, 6]

#numpy 안에서는 [5,7,9] 이런 식으로 사람이 하는 계산처럼 직관적으로 나온다 np.array[]

c = [7,8,9,10] 
print(a+c) # [1, 2, 3, 7, 8, 9, 10]

print (a*3) # [1, 2, 3, 1, 2, 3, 1, 2, 3]



# print (a[2] + 'hi')  #error

print (str(a[2]) + 'hi')     #3hi

f = '5'
# print ((a[2]) + f) #error
print ((a[2]) + int(f)) #8 

# 리스트 관련 함수

#진짜 가장 많이 보게 되는 함수

'''

*append* 중요 
sort
index
insert 
remove


'''


a.append(4)  #덧붙임
    
    # a = a.append(5)  #error
    # 그냥 a.append() 그 자체

a = [1, 3, 4, 2]
a.sort()
print(a)  # [1, 2, 3, 4]

a.reverse() 
print(a)  # [4, 3, 2, 1]

print(a.index(3))  # [4, 3, 2, 1] 상태니까 1 이 나온다 * == a[3]
print(a.index(1))  # * == a[1]

a.insert(0, 7) 
print(a)  # [7, 4, 3, 2, 1]

a.insert(3, 3)
print(a) # [7, 4, 3, 3, 2, 1]

a.remove(7) 
print(a) # [4, 3, 3, 2, 1]

a.remove(3)
print(a)  # [4, 2, 1]? [4, 3, 2, 1]? -> 먼저 걸린 놈만 지워진다 [4, 3, 2, 1] 


'''

list, slicing, append 는 주구장창 쓸 거다 


'''