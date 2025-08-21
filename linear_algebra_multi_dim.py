from typing import List, Tuple, Callable
import math

### example 1
Matrix = List[List[float]] # type alias

def shape(A: Matrix) -> Tuple[int, int]:
    num_row = len(A)
    num_column = len(A[0])
    return (num_row, num_column)

def get_row(A: Matrix, row: int) -> List[float]:
    return A[row]

def get_column(A: Matrix, column: int) -> List[float]:
    return [A[i][column] for i in range(len(A))]

def create_matrix(num_row: int, num_column: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j) for j in range(num_column)] for i in range(num_row)]

def identity_matrix(n: int) -> Matrix:
    return create_matrix(n, n), lambda i, j: 1 if i == j else 0

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
assert get_column([[1, 2, 3], [4, 5, 6]], 2) == [3, 6]
assert get_row([[1, 2, 3], [4, 5, 6]], 1) == [4, 5, 6]
assert create_matrix(2, 2, lambda i, j: 1 if i == j else 0) == [[1, 0], [0, 1]]
assert create_matrix(3, 3, lambda i, j: 1 if i == j else 0) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
