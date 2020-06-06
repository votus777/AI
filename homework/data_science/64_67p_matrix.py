'''

Matrix = List[List[float]]

A = [[1,2,3,],[4,5,6]]

B = [[1,2,], [3,4],[5,6]]


from typing import Tuple

def shape(a:Matrix) -> Tuple[int,int]:

    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

assert shape([[1,2,3],[4,5,6]]) == (2,3) 


def get_row(A: Matrix, i:int) => Vector :   
    return A[i]

def get_column(a:Matrix, j: int) -> Vector :
    return [A_i[j] for A_i in A]
   

from typing import Callable 


def make_matrix(num_row: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix

return  [[emtry_fn(i,j) for j in range (num_cols)]
            for i in range(num_rows)]


def identity_matrix(n:int) -> Matrix :
    return make_matrix (n,n, lambda i , j :1 if i == j else 0)

assert identity_matrix(5) == [[1,0,0,0,0], [0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,1]]

date = [[70,170,40],[65,120,26],[77,250,19]]

friendships = [ (0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,8),(7,8),(8,9)]

friend_matrix = [0,1,1,0,0,0,0,0,0],[1,0,1,1,0,0,0,0,0]

assert friend_matrix[0][2] == 1,
assert friend_matrix[0][8] ==0

friends_of_five = [i for i, is_friend in enumerate(firend_matix[5]) if is_firend]

'''