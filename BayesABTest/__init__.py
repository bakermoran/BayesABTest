"""
BayesABTest python package.

A simple to use interface for Bayesian
hypothesis testing. Meant to provide analysts and data scientists with an
abstracted way to quickly test a hypothesis. Provides a uniform reporting
chart for test results.
"""

from . import priors
from .ab_test import ABTest
from .ab_test_dist_explorer import beta_dist, gamma_dist, lognormal_dist, poisson_dist
from .result import LossResult, Posterior, Result

__all__ = ['ABTest', 'Result', 'Posterior', 'LossResult', 'priors',
           'beta_dist', 'gamma_dist', 'lognormal_dist', 'poisson_dist']
