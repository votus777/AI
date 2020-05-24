#112p

from scratch.linear_algebra import Vector, dot

def sum_of_square(v: Vector) -> float :

    return dot(v,v)


#113p

from typing import Callable

def difference_quotient (f: Callable[[float], float],
                            x: float,
                            h: float) -> float :

    return (f(x+h) - f(x) /h)


#114p

 def square(x):
        return x * x

def derivative(x):
        return 2 * x


 x = range(-10, 10)
    plt.title("Actual Derivatives vs Estimates")
    plt.plot(x, map(derivative, x), 'rx', label='Acutal')           
    plt.plot(x, map(derivative_estimate, x), 'b+', label='Estimate')  
    plt.legend(loc=9)                                              
    plt.show()        


#115p

def partial_difference_quotient(f: Callable[Vector], v:Vector, i:int, h:float =0.0001):

       w = [v_j + (h if j == i else 0)    
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):

    return [partial_difference_quotient(f: Callable[Vector], v:Vector, i:int, h:float =0.0001):
            for i,  in range(len(v))]
        

#116p

import random 
from scratch.linear_algebra import distance, add, scalar_multiply

assert len(v) == len(gradient)
step = scalat_multiply(step_size, gradient)
return add (v, step)


def sum_of_squares_gradient( v: Vector)-> Vector :
    return [ 2 * v_i for v_i in v]


v = [random.uniform(-10,10) for i in range (3) ]

for epoch in range(1000) :
        grad = sum_of_squares_gradient(v) 
        v= gradient_step(v, grad, -0.01)
        print(epoch, v)


#117p
def linear_gradient(x: float, y: float, theta : Vector) -> Vector :
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (pridicted -y)
    squared_error = error ** 2
    grad = [ 2* error * x, 2 * error]
    return grad

from scratch.linear_algebra import vector_mean

theat = [random.uniform(-1,1), random.uniform(-1,1)]

learning_rate = 0.001

for epoch in range(5000) :
    grad = vector_mean([lineat_gradient(x,y,theta) for x,y in inputs])

    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch. theta)

slope, intercept = theta
assert 19.9 < slope < 20.1
assert 4.9 < intercept < 5.1 

#119p

from typing import TypeVar, Lsit, Iterator

T = TypeVar('T') 


def minibatches(dataset: List[T],  batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
 
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)  

    for start in batch_starts:
        end = start + batch_size

        yield dataset[start:end]

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    learning_rate = 0.001
    
    for epoch in range(5000):
       
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta

    assert 19.9 < slope < 20.1,   
    assert 4.9 < intercept < 5.1


theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    for epoch in range(1000):
        
        for batch in minibatches(inputs, batch_size=20):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)
    
    slope, intercept = theta


assert 19.9 < slope < 20.1 
assert 4.9 < intercept < 5.1
