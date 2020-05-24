
#88p

import enum, random

#Enum을 사용하면 각 항목에 특정 값을 부여할 수 있으며
#파이썬 코드를 더욱 깔금하게 만들어 줌

class Kid(enum.Enum) :
    BOY =0
    GIRL = 1

def random_kid() -> Kid :
    return random.choice([Kid.BOY, Kid.GIRL])


both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range (10000) :
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL :
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL :
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL :
        either_girl += 1

print("P(both | older):", both_girls / older_girl)
print("P(both | iether):", both_girls / either_girl) 


# 89p

def uniform_pdf(x:float) -> float :
    return 1 if 0 == x< 1 else 0

def uniform_cdf(x:float) -> float :
    if x<0 : return 0
    elif x < 1 : return x
    else : return 1



# 90p

import math
SQRT_TWO_PI = math.sqrt(2*math.pi)

def noraml_pdf(x: float, mu : float = 0 , sigma : float =1) -> float :
    return (math.exp(-(x-mu) ** 2/2/sigma ** 2) / (SQRT_TWO_PI * sigma))

import matplotlib.pyplot as plt








