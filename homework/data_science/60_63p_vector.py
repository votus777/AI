from typing import List

Vector = List[float]

height_weight_age = [70, 170, 40]

grade  = [95, 80, 75, 62]

def add (v:VEctor, w: Vector) -> Vector :

    assert len(v) == len(w), "vectors must be the same length"

    return [ v_i + w_i for v_i, w_i in zip(v,w)]


def substact(v: Vector, w:Vector) -> Vector :

    assert len(v) == len(w), "vectors must be the same length"

    return [ v_i - w_i for v_i, w_i in zip(v,w)]



def vector_sum(vectors:List[Vector]) -> Vector :

    assert vectors, "no vectors provided!"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    return [sum(vector[i] for vector in vectors)]
            # for i in range(num_elements)


assert vector_sum([[1,2],[3,4],[5,6],[7,8]]) == [16,20]

def scalar_multiply(c: float, v:Vector) -> Vector :

    return [ c*v_i for v_i in v]

assert scalar_multiply(2, [1,2,3]) == [2,4,6]


def vector_mean (vectors : List[Vector]) -> Vector :

    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1,2],[3,4],[5,6]]) == [3,4]

def dot (v: Vector, w:vector) -> float :
    """v_1 * w_1 + ... + v_n  * w_n"""

    assert len(v) == len (w), "vectors must be same length"

    return sum (v_i * w_i for v_i,w_i in zip(v,w))

assert dot([1,2,3,], [4,5,6]) == 32

