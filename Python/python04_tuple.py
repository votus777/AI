

'''

2. 튜플

리스트와 거의 같으나, 삭제, 수정이 안된다.  고정값

변경이 안되는 값

'''

a = (1,2,3)
b = 1, 2, 3
c = [1, 2, 3]


print(type(a)) # class tuple
print(type(b)) # class tuple 
print(type(c)) # clas  list

# a.remove(2) 
# print(a) error : AttributeError: 'tuple' object has no attribute 'remove'

print( a + b )  # (1, 2, 3, 1, 2, 3)
print ( a * 3 ) # (1, 2, 3, 1, 2, 3, 1, 2, 3)
# print ( a + 3 ) # error : TypeError: can only concatenate tuple (not "int") to tuple

