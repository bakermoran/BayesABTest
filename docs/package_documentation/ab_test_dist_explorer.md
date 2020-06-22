# Class ab_test_dist_explorer

This file contains a set of functions to help a user visualize each distribution.

## Methods

* `beta_dist(alpha, beta)`
  * *requires* - alpha and beta are non-negative integers. alpha signifies the successes, and beta signifies the failures. ex: alpha = 10, beta = 10. Sucess rate = 50% out of a population of 20.
  * *modifies* - returns a matplot lib object (show the plot with `matplotlib.pyplot.show()`)
  * *effects* - none
* `gamma_dist(mean=None, var=None, alpha=None, beta=None)`
  * *requires* - Inputs are in pairs, either [mean, variance] OR [alpha, beta]. Mean is on the inveval (-inf, inf), variance is greater than 0. Alpha and beta are both postive numbers.
  * *modifies* - returns a matplot lib object (show the plot with `matplotlib.pyplot.show()`)
  * *effects* - none
* `lognormal_dist(mean, var)`
  * *requires* - mean and variance of the log-normal distribution. Mean is on the inveval (-inf, inf), variance is greater than 0.
  * *modifies* - returns a matplot lib object (show the plot with `matplotlib.pyplot.show()`)
  * *effects* - none
* `poisson_dist(lam)`
  * *requires* - lambda is the average occurences per unit time in a poisson distribution. Lambda is non-negative.
  * *modifies* - returns a matplot lib object (show the plot with `matplotlib.pyplot.show()`)
  * *effects* - none

## Usage Guide

```python
# import packages
from BayesABTest import ab_test_dist_explorer as pe
from matplotlib import pyplot as plt

# use the functions to plot a distribution
pe.beta_dist(20, 80)
plt.show()
```

## Examples

### Beta(20, 80) Distribution

![alt text](img/beta_20_80.png "Beta(20, 80) Distribution")

### Gamma(4, 2) Distribution

![alt text](img/gamma_4_2.png "Gamma(4, 2) Distribution")

### Poisson(15) Distribution

![alt text](img/poisson_15.png "Poisson(15) Distribution")
