# for 반복문

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}

for i in a.keys() :                     # i에 순서대로 키 값을 넣어준다
# : 이후 enter -> tab 이 되면서 밑에서 부터 상위문에 속하게 된다.
    print(i)                            


'''

결과값

name
phone
birth

'''

a = [1,2,3,4,5,6,7,8,9,10]
for i in a :
    i = i*i
    print(i)

'''
for i in a :
    i = i*i
    print(i)
print(abc)    -> abc 한 번 출력   (for문 밖에 있음)


for i in a :
    i = i*i
    print(i)
    print(abc) -> abc 10번 출력   (for문 안에 있음)
'''

'''
1
4
9
16
25
36
49
64
81
100

'''


## while문 

''' 
while 조건문 :               #조건문이 참일 동안 계속 돈다. 
     수행할 문장

'''

### if문 

if 1:
    print('True')
else :
    print('False')

#True가 나온다. 



if 3:
    print('True')
else :
    print('False')
#True가 나온다 


if 0:
    print('True')
else :
    print('False')
#False가 나온다


if -1:
    print('True')
else :
    print('False')
#True가 나온다 

'''

비교연산자

<, >, ==, !=, <=, >=,

'''

# if a = 1:
    # print(출력)   -> error()

if a == 1:
    print('출력')  # -> 출럭


money = 10000
if money >= 30000:
    print('한우 먹자')
else : 
    print('라면 먹자')   #라면 먹자



### 조건연산자 

# and, or, not 

money = 20000
card = 1
if money >= 30000 or card == 1 :
    print('한우 먹자')
else :
    print('라면 먹자') #한우 먹자

#_________________________________________

#break 
# break :  걸릴 시 for 그 문장에서 가장 가까운 반복문을 중지시킨다 

score  = [ 90, 25, 67, 34, 60]
number = 0
for i in score :
    if i < 30:
        break

    if i >= 60 :
        print("합격")
        number = number + 1 
print("합격 인원 :", number, "명") #합격, 합격 인원 : 1 명



# continue
score  = [ 90, 25, 67, 34, 60]
number = 0
for i in score :
    if i < 60:
        continue

    if i >= 60 :
        print("합격")
        number = number + 1 
print("합격 인원 :", number, "명")  # 합격 합격 합격 합격 인원 : 3 명

# continue 걸리면 바로 위 for문으로 돌아감 

