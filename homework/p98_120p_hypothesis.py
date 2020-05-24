# 99p
from typing import Tuple
import math

def normal_apporximation_to_binomial(n: int, p: float) -> Tuple [float, float] :
    mu = p*n
    sigma = math.sqrt(p*(1-p) * n)
    return mu, sigma


from scratch.probability import normal_cdf

noraml_probability_below = normal_cdf

def normal_probability_above(lo : float, mu: float =0, sigma : float =1) -> float : 

    return 1 - normal_cdf (lo, mu, sigma)

def normal_probability_between (lo :float, hi : float, mu: float =0, sigma: float =1) -> float:

    return normal_cdf (hi,mu,sigma) - normal_cdf(lo,mu,sigma)


def normal_probability_outside(lo : float, hi : float, mu : float =0, sigma : float =1) -> float :

    return 1 - normal_probability_between(lo,hi,mu,sigma)


from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability:float, mu : float =0, sigma:float =1) -> float:

    return inverse_normal_cdf(probability, mu, sigma )


def normal_two_sided_bounds(probability : float, mu : float =0, sigma : float =1) -> Tuple[float,float] ::

    tail_probability = ( 1- probability) /2 

    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    lower_bound= normal_upper-bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound



# 100p

mu_0, sigma_0 = normal_approximation_to_binomial(1000,0.5)

lower_bound, upper_bound = noraml_two_sided_bounds (0.95, mu_0, sigma_0)

lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)

lo,hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

mu_1,sigma_1 = normal_apporximation_to_binomial(1000, 0.55)

type_2_probability = normal_probability_between(lo, hi , mu_1, sigma_1)


#101p

hi = normal_upper_bound(0.95, mu_0, sigma_0)

type2_probability = noraml_probability_below(hi,mu_1, sigma_1)
power = 1 - type_2_probability


def two_sided_p_value(x: float, mu:float =0, sigma : float =1) -> float :

    if x >= mu:
        return 2 * normal_probability_above(x,mu,sigma)

    else : 
        return 2 * noraml_probability_below(x,mu,sigma)


two_sided_p_value(529.5, mu_0, sigma_0)


#102p

import random
extreme_value_count = 0

for _ in range (1000) :
    num_heads = sum (1 if random.random () < 0.5 else 0 
    for _ in range (1000))

    if num_head >= 530 or num_head <= 470 :
        extreme_value_count += 1


two_sided_p_value(531.5, mu_0, sigma_0)

upper_p_value = normal_probability_above
lower_p_value = noraml_probability_below

upper_p_value(524.5, mu_0, sigma_0 )


upper_p_value (526.5, mu_0, sigma_0)


# 103p

math.sqrt(p*(1-p) /1000)

p_hat = 525 / 1000
mu = p_hatsigma = math.sqrt (p_hat *( (1- p_hat)/1000)

normal_two_sided_bounds(0.95, mu, sigma)


#104p 

from typing import List

def run_experiment() -> Lsit [bool]:

    return [ random,random() < 0.5 for _in range(1000)]


def run_experiment():
        return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment):

    num_heads = len([flip for flip in experiment if flip])

    return num_heads < 469 or num_heads > 531



# 106p

def estimated_parameters(N, n):
    p = n / N

    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A, n_A, N_B, n_B):

    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

#107p

def B(alpha, beta):
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:      return 0        
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)

