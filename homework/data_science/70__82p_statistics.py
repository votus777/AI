from collections import Counter
import matplotlib.pyplot as plt


num_friends = [100,23, 46, 23, 34, 47]
List = [[100,23], [46, 23], [34, 47]]


friend_counts = Counter(num_friends)

xs = range(101)
ys = [friend_counts[x] for x in xs]
plt.bar (xs,ys)
plt.axis([0,101,0,25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of firends")
plt.ylabel("# of people")
plt.show()

num_points = len(num_friends)

largest_value = max(num_friends)
smallest_value = min(num_friends)


sorted_values = sorted(num_friends)
smallest_value = sorted_values[1]
second_largest_value = sorted_values[-2]

def mean(xs: List[float]) -> float :
    return sum(xs) / len(xs)

mean(num_friends)


def _median_odd(xs: List[float]) -> float :
    return sorted(xs)[len(xs) //2]

def _median_even(xs : List[float]) -> float :
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) //2
    return (sorted_xs[hi_midpoint -1] + sorted_xs[hi_midpoint]) /2

def median(v:List[float]) -> float :
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

assert median ([1,10,2,9,5]) == 5
assert median ([1,9,2,10]) == (2+9) /2

print(median(num_friends))

def quantile(xs : List[float], p : float) -> float : 
    p_index = int(p*len(xs))
    return srted(xs)[p_index]

assert quantile(num_friends, 0.10) ==1
assert quantile(num_friends, 0.25) ==3
assert quantile(num_friends, 0.75) ==9
assert quantile(num_friends, 0.90) ==13

def mode(x:List[float]) -> List[float] :
    counts = Counter(x)
    max_cound = max (counts.values())
    return [x_i for x_i, count in counts.items()
    if count == max_xount]

assert set (mode(num_firends)) == {1,6}

#74p
def date_range (xs : List [float]) -> float :
    return max(xs) - min(xs)

assert date_range(num_friends) == 99




from scratch.linear_algebra import sum_of_squares

def de_mean(xs : List[float]) -> List[float]:
    x_bar = mean(xs)
    return [ x-x_bar for x in xs]

def varivance(xs: List[float]) -> float :
    assert len(xs) >=2, "variance requires at leat rwo elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n -1)

assert 81.54 < variance (num_driends) < 81.55



#75p
import math 

def standard_deviation(xs: List[float]) -> float :
    return math.sqry(variance(xs))

assert 9.02 < standard_deviation(num_friends) < 9.04

def interquartile_range(xs: List[float]) -> float :
    return quantile(xs, 0.75) - quantile(xs, 0.25)

assert interquatile_range(num_friends) == 6




# 76p
from scratch.linear_algebra import dot

def covariance(xs:List[float], ys: List[float]) -> float :
    assert len(xs) == len(ys), "xs and ys must have same number of elements"

    return dot (de_mean(xs), de_mean(ys)) / len(xs) -1)

assert 22.42 < covariance(num_friends, daily_minutes) < 22.43
assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60


#77p
def correlation(xs: List[float], ys: List[float]) -> float :

    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_X > 0 and stdev_y > 0:
        return covariance (xs,ys) / stdev_X / stdev_y

assert 0.24 < correlation(num_friends, daily_minutes) < 0.25
assert 0.24 < correlation(num_friends, daily_hours) < 0.25


#78p
outlier = num_friends.index(100) 

num_firends_good = [x for i, x in enumerate(num_friends) if i ! = outlier]

daily_minutes_good = [ x for i, x in enumerate(num_minutes) if i ! = outlier]

daily_hours_good = [dm/ 60 for dm in daily_minutes_good]

assert 0.57 < correlation(num_friends_good, daily_minutes_good) < 0.58
assert 0.57 < correlation(num_friends_good, daily_hours_good) < 0.58



