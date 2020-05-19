
# 정수형 

a = 1
b = 2
c = a + b

# print(c) 
# a = 3

d = a* b

print (d) 

# d = 2 


e = a / b 
print(e) 

# e = 0.5 
# C 나 java 에서는 1 / 2 하면 0이 나오지만 pyhton 에서는 알아서 잘 나온다 

# 실수형 
a = 1.1
b = 2.2 

c - a+b
print(c)
#  c = 3

d = a*b 
print(d) 

# d = 2.420000000000004  <- ??? 파이썬 부동 소수점 오차, 실수 연산의 끝에서 항상 틀어짐 

c = a/b 
print(c)

#c = 0.5


#문자형 
a = "hel"
b = "lo"
c =  a + b 

print(a+b) 
# hello 


a = 123
b = "45" 


# print(c)
# errror
# TypeError: unsupported operand type(s) for +: 'int'(정수) and 'str'(문자열)

#  추후에 데이터 받을 때 그 데이터가 모두 숫자란 보장 X 

#  1. 숫자를 문자로 변환 + 문자 

a = 123
a = str (a) 
print (a) 

b= '45'
c = a + b 
print(c)  

# 2. 문자를 숫자 변환

a= 123
b = "45"

b = int(b) 

c = a + b
print(c) 
#  168 = 123 + 45

#  형변환 

#  문자열 연산하기 

a = 'abcdefgh' 
print(a[0]) #a

print(a[3]) #d 

print(a[5]) #f 

print(a[-1]) #h 

print(type(a)) #str

b = 'xyz'
print ( a + b) #abcdefghxyz

# 문자열 인덱싱 

a = 'Hello, Deep learning' 

#  ",(쉼표)", "공백" : 모두 문자 
# 시작 인덱스는 항상 0 이다 

print(a[7]) # D
print(a[-1]) # g
print(a[-2]) # n
print(a[3:9]) # lo, De
print(a[3:-5]) # lo, Deep lea
print(a[:-1]) #Hello, Deep learnin 
print(a[1:]) #ello, Deep learning  
print(a[5:-4]) # , Deep lear

