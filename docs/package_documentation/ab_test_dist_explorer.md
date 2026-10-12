# Module ab_test_dist_explorer

This file contains a set of functions to help a user visualize each distribution.

## Functions

Each function plots on `ax` (a keyword argument, defaulting to the current Axes) and returns the frozen `scipy.stats` distribution, so you can also query it (`.mean()`, `.ppf(0.95)`, ...). Show the plot with `matplotlib.pyplot.show()`.

* `beta_dist(alpha, beta, *, ax=None)`
  * `alpha` and `beta` are positive. alpha signifies the successes, and beta signifies the failures. ex: alpha = 10, beta = 10. Success rate = 50% out of a population of 20.
* `gamma_dist(mean=None, var=None, alpha=None, beta=None, *, ax=None)`
  * Inputs are in pairs, either [mean, variance] OR [alpha, beta] (beta is the rate). All are positive.
* `lognormal_dist(mean, var, *, ax=None)`
  * The mean and variance of the log-normal distribution itself (not of its logarithm). Both are positive.
* `poisson_dist(lam, *, ax=None)`
  * lambda is the average occurrences per unit time in a poisson distribution. Lambda is positive.

## Usage Guide

```python
# import packages
from BayesABTest import ab_test_dist_explorer as pe
from matplotlib import pyplot as plt

# use the functions to plot a distribution
dist = pe.beta_dist(20, 80)
print(dist.mean())  # 0.2
plt.show()
```

## Examples

### Beta(20, 80) Distribution

![alt text](img/beta_20_80.png "Beta(20, 80) Distribution")

### Gamma(alpha=8, beta=2) Distribution

Plotted with `gamma_dist(mean=4, var=2)`, which is the same distribution.

![alt text](img/gamma_4_2.png "Gamma(alpha=8, beta=2) Distribution")

### Poisson(15) Distribution

![alt text](img/poisson_15.png "Poisson(15) Distribution")
