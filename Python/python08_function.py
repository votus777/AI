

def sum1 (a,b) :
    return a + b


print(sum1(3,4)) #7

a = 1
b = 2
c = sum1(a,b)
print(c)  # 3 

def mul1 (a,b) :
    return a*b

print(mul1(a,b))   #2

def div1 (a,b):
    return a%b
print(div1(a,b)) #1



def sub1 (a,b) :
    return a - b 
print(sub1(a,b)) #-1

def sayYeaaaaah () : 
    return 'Yeahh'
    
aaa = sayYeaaaaah ()
print(aaa)   # Yeahh

# 매개변수(parameter)가 없어도 함수값 출력 

def sum2 (a,b,c) :
    return a + b + c

a = 1
b = 2
c = 3

d = sum2(a,b,c)

print(d)