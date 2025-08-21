from collections import Counter
import matplotlib.pyplot as plt
from typing import List
from linear_algebra_one_dim import sum_of_squares, dot
import math

num_friends = [100, 49, 41, 40, 25, 42, 30, 28, 15, 9, 50, 11, 4, 47]
friend_counts = Counter(num_friends)

xs = range(15)
ys = [friend_counts[x] for x in xs]
plt.bar(xs, ys)
plt.axis([0, 15, 0, 50])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
# plt.show()


## 統計數字
num_points = len(num_friends)
largest_value = max(num_friends)
smallest_value = min(num_friends)

sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]
second_smallest_value = sorted_values[1]
second_largest_value = sorted_values[-2]
largest_value = sorted_values[-1]

## mean
def data_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

resMean1 = data_mean(num_friends)
print("mean1 res = " + str(resMean1))

## median
def _data_median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs) // 2]

def _data_median_even(xs: List[float]) -> float:
    sorted_list = sorted(xs)[len(xs) // 2]
    higher_index = len(xs) / 2
    return (sorted_list[higher_index - 1] + sorted_list[higher_index]) / 2

def data_median(xs: List[float]) -> float:
    return _data_median_even(xs) if len(xs) % 2 == 0 else _data_median_odd(xs)

def data_median_quick_select(xs: List[float]) -> float:
    # TODO
    return 0.0

## outlier -> quantile 
def data_quantile(xs: List[float], p: float) -> float:
    pp = 1 if p < 0 or p > 1 else p
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

## mode 眾數
def data_mode(xs: List[float]) -> List[float]:
    counter = Counter(xs) 
    max_count = max(counter.values())
    return [x for x, count in counter.items() if count == max_count]

## 數據的離散程度 (dispersion) 
### range 範圍值來計算
### 1. 計算最大及最小值的差值
def dispersion_range_max_min(xs: List[float]) -> float:
    return max(xs) - min(xs)

### 2. 計算75分位數及25分位數的差值
def dispersion_range_75_25(xs: List[float]) -> float:
    value_75 = data_quantile(xs, 0.75)
    value_25 = data_quantile(xs, 0.25)
    return value_75 - value_25

### variance 變異數, standard deviation 標準差來計算
def data_variance(xs: List[float]) -> float: 
    mean = data_mean(xs)
    diffs_mean = [x - mean for x in xs ]
    return sum_of_squares(diffs_mean) / (len(xs) - 1)

def data_standar_deviation(xs: List[float]) -> float:
    return math.sqrt(data_variance(xs))

## 相關程度 
### covariance 共變異數 
def data_covariance(xs1: List[float], xs2: List[float]) -> float:
    if len(xs1) != len(xs2): return 0.0

    mean1 = data_mean(xs1)
    mean2 = data_mean(xs2)
    diffs_mean1 = [x - mean1 for x in xs1]
    diffs_mean2 = [x - mean2 for x in xs2]
    return dot(diffs_mean1, diffs_mean2) / (len(xs1) - 1)

### correlation 相關係數
def data_correlation(xs1: List[float], xs2: List[float]) -> float:
    standar_deviation1 = data_standar_deviation(xs1)
    standar_deviation2 = data_standar_deviation(xs2)

    if standar_deviation1 != 0 and standar_deviation2 != 0:
        return data_covariance(xs1, xs2) / (standar_deviation1 * standar_deviation2)
        return data_covariance(xs1, xs2) / data_variance(xs1)
    else: 
        return 0