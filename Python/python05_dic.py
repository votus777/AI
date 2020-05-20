 #3. 딕셔너리 # 중복 X 
 # {key : value}

a = {1 : 'hi', 2: 'Hello'}
print(a) #{1: 'hi'}
print(a[1]) #hi

b = {'hi' : 1, 'Hello' : 2}
print(b['Hello']) #2

# 딕셔너리 요소 삭제 

del a[1] 
print(a)  #{2: 'Hello'}

del a[2]
print(a) #{}


c = {1: 'a', 2: 'b', 1:'b', 1: 'c'}
print(c) #{1: 'c', 2: 'b'}   # 중복된 키값이  덮어씌워져서 마지막 value가 나온다 
print(c[1]) #c


d = {1:'a', 2: 'a', 3:'a'}
print(d) #{1: 'a', 2: 'a', 3: 'a'} # 이건 다 나온다

e = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}

print(e.keys()) #dict_keys(['name', 'phone', 'birth'])
print(e.values()) #dict_values(['yun', '010', '0511'])
print(type(e)) #<class 'dict'>
print(e.get('name')) #yun 
print(e['name']) #yun 
print(e.get('phone')) #010
print(e['phone']) #010 



