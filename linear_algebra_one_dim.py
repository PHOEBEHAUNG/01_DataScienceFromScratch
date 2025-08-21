from typing import List
import math

Vector = List[float] # type alias
height_weight_age = [70, 170, 40]
grade = [95, 80, 75]

print(zip(height_weight_age, grade))

def add(v: Vector, w: Vector) -> Vector:
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    # 1. vector 向量不為 None
    # 2. vectors 長度一致

    if vectors == None:
        return None
    
    len_vec = len(vectors[0])
    for v in vectors:
        if len_vec != len(v):
            return None

    return [sum(vector[i] for vector in vectors) for i in range(len_vec)]

def scalar_multiply(c: float, vector: Vector) -> Vector:
    return [c * v for v in vector]

def vector_mean(vectors: List[Vector]) -> Vector:
    if vectors == None:
        return None
    len_vec = len(vectors)

    return scalar_multiply(1 / float(len_vec), vector_sum(vectors))

'''
相應元素 相乘總和
'''
def dot(v: Vector, w: Vector) -> float:
    if len(v) != len(w):
        return None
    # return sum(v_i * w_i for v_i, w_i in zip(v, w))
    return sum(v[i] * w[i] for i in range(len(v)))

def sum_of_squares(v: Vector) -> float:
    return sum(v[i] * v[i] for i in range(len(v)))

'''
向量長度 向量大小
(a ^ 2 + .... + z ^ 2) ^ 0.5
'''
def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))

'''
兩向量相減後的平方和
'''
def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    return math.sqrt(squared_distance(v, w))

def distance2(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))


# Test 
assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
assert scalar_multiply(3, [1, 2, 3]) == [3, 6, 9]
assert vector_mean([[3, 2], [3, 4], [5, 6], [7, 8]]) == [4.5, 5]
assert dot([1, 2, 3], [4, 5, 6]) == 32
assert sum_of_squares([1, 2, 3]) == 14
assert magnitude([3, 4]) == 5