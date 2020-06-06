from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float] :
    mu = p*n
    sigma = math.sqrt (p* (1-p) * n)
    return mu, sigma


from scratch.probability import normal_cdf