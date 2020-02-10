import pandas as pd
import numpy as np
import scipy.special as sc
from scipy import stats
from BayesABTest import BayesABTest as ab
#pylint: disable=no-member

# REFERENCE: https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html
# https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf

# "This stopping condition considers both the likelihood that β — α
# is greater than zero and also the magnitude of this difference.
# https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5

# currently only analytical solution for a one variant beta test is implemeneted
# will have to numerically do other ones

def h(a,b,c,d):
  sum = 0
  for j in range(0,c):
    w = sc.betaln(a+j, b+d)
    x = np.log(d + j)
    y = sc.betaln(1 + j, d)
    z = sc.betaln(a,b)
    sum += np.exp(w-x-y-z)
  print(sum)
  return 1 - sum

def loss_beta(a,b,c,d, type='absolute'):
  if type == 'absolute':
    x1 = sc.betaln(a+1,b)
    y1 = np.log(h(a+1,b,c,d))
    z1 = sc.betaln(a,b)

    x2 = sc.betaln(c+1,d)
    y2 = np.log(h(a,b,c+1,d))
    z2 = sc.betaln(c,d)

    return np.exp(x1 + y1 - z1) - np.exp(x2 + y2 - z2)

  elif type == 'percent':
    prob_1 = h(a,b,c,d)

    x = sc.betaln(a - 1, b)
    y = sc.betaln(a, b)
    z = sc.betaln(c + 1, d)
    w = sc.betaln(c, d)
    prob_2 = np.log(h(a - 1, b, c + 1, d))

    return prob_1 - np.exp(x - y + z - w + prob_2)

def loss_gamma(a,b,c,d):
  x1 = sc.gammaln(a+1)
  y1 = np.log(h(a+1,b,c,d))
  z1 = np.log(b)
  w1 = sc.gammaln(a)

  x2 = sc.gammaln(c+1)
  y2 = np.log(h(a,b,c+1,d))
  z2 = np.log(d)
  w2 = sc.gammaln(c)

  return np.exp(x1 + y1 - z1 - w1) - np.exp(x2 + y2 - z2 - w2)

def loss_normal(sample1, sample2):
  diffs = []
  for x, y in zip(sample1, sample2):
    diffs.append(max(x-y), 0)
  return np.mean(diffs)


def stop_test(risk_A, risk_B, epsilon=.0001):
  if risk_A <= epsilon or risk_B <= epsilon: return True
  else: return False



# TEST DATA
prior_alpha = 2
prior_beta = 7

alpha_likelihood_A = int(.27 * 1000 + prior_alpha)
beta_likelihood_A = int(1000 - .27 * 1000 + prior_beta)

alpha_likelihood_B = int(.3 * 1000 + prior_alpha)
beta_likelihood_B = int(1000 - .3 * 1000 + prior_beta)

# OUTPUT
print('absolute choosing A', loss_beta(alpha_likelihood_B, beta_likelihood_B, alpha_likelihood_A, beta_likelihood_A))
print('absolute choosing B', loss_beta(alpha_likelihood_A, beta_likelihood_A, alpha_likelihood_B, beta_likelihood_B))

print('percent choosing A', loss_beta(alpha_likelihood_B, beta_likelihood_B, alpha_likelihood_A, beta_likelihood_A, type='percent'))
print('percent choosing B', loss_beta(alpha_likelihood_A, beta_likelihood_A, alpha_likelihood_B, beta_likelihood_B, type='percent'))
