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