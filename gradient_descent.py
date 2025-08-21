from typing import Callable
from typing import List
from typing import TypeVar, List, Iterator

# epoch : 階段
# average squared error

Vector = List[float] # type alias

# 導數: derivative -> 經過 x 的切線斜率
# 差商: difference quotient -> x, x + h 的割線斜率
def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float: 
    return (f(x + h) - f(x)) / h

# 計算第 i 個變數的偏導數
def partial_difference_quotient(f: Callable[[Vector], float], v: Vector, i: float, h: float) -> float: 
    w = [v_j + (h if j == i else 0) for j, v_j in v]
    return (f(w) - f(v)) / h

# 利用差商來評估對於計算能力的需求很高
def estimate_gradient(f: Callable[[Vector], float], v: Vector, h: float = 0.0001) -> Vector:
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

import random
from linear_algebra_one_dim import distance, add, scalar_multiply, vector_mean

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    gradient_step = scalar_multiply(step_size, gradient)
    return add(v, gradient_step)

def double_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# 梯度應用
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]
v = [random.uniform(-10, 10) for i in range(3)]
for epoch in range(1000):
    grad = double_gradient(v)
    v = gradient_step(v, grad, -0.01)
    print(epoch, v)

# 梯度間隔的選擇
# 1. 使用固定間隔
# 2. 隨時間逐漸縮小間隔
# 3. 每個步驟都重新選擇"讓目標函數值最小化"的尖閣
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = predicted - y
    squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad 

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
learning_rate = 0.001
for epoch in range(5000):
    # theta 一直變，去調整模型
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1
assert 4.9 < intercept < 5.1


#### 小批量梯度遞減 (mini)batch gradient descent
T = TypeVar('T')

def minibatches(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle: 
        random.shuffle(batch_starts)
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]        
    return batch_starts

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
for epoch in range(1000):
    # theta 一直變，去調整模型
    for batch in minibatches(inputs, batch_size = 20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1
assert 4.9 < intercept < 5.1

#### 隨機梯度遞減 stochastic gradient descent
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
for epoch in range(100):
    # theta 一直變，去調整模型
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1
assert 4.9 < intercept < 5.1