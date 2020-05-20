
# range 클래스 

a = range(10) 

print(a) # range(0,10)

b = range(1,11)
print(b)  # range(1,11)

for i in a :
    print(i) # 0,1,2,3,4,5,6,7,8,9


''' 
range 함수는 마지막-1까지 나온다  

'''

print(type(a))  # class 'range'

sum =0

for i in range (1,11) :
    
    sum = sum + i  
print(sum)  # 55 
