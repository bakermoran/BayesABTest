"""Fucntions to help explore distribution parameters."""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.special import factorial


def beta_dist(alpha, beta):
    """Plot a Beta distribution.

    Arguments:
        alpha {int} -- alpha parameter to beta function. Represents the amount
            of conversions. must be greater than 0.
        beta {int} -- beta parameter to beta function. Represents the amount
            of non-conversions. must be greater than 0.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError('alpha and beta must both be greater than 0.')

    x = np.linspace(0, 1, 1000)
    dist = stats.beta(alpha, beta)
    y = dist.pdf(x)
    sns.lineplot(x, y, color='green')
    plt.fill_between(x, y, color='green', alpha=.2)
    sns.despine(left=True)
    plt.yticks([], [])
    plt.title('Beta({0}, {1}) Distribution PDF'.format(alpha, beta),
              fontweight='bold', fontsize=14)
    plt.xlabel('Conversion Rate')
    plt.xlim((0, 1))
    locs, labels = plt.xticks()
    labels = []
    for i in range(len(locs)):
        labels.append('{:.0%}'.format(locs[i]))
    plt.xticks(locs, labels=labels)


def gamma_dist(mean=None, var=None, alpha=None, beta=None):
    """Plot a Gamma distribution.

    Pick either (mean and var) or (alpha and beta)
    Arguments:
        mean {int} -- mean parameter to gamma distribution.
        var {int} -- var parameter to gamma distribution.
            Must be greater than 0.
        alpha {int} -- alpha parameter to gamma distribution.
            Must be greater than 0.
        beta {int} -- beta parameter to gamma distribution.
            Must be greater than 0.
    """
    if any(i == 0 for i in [mean, var, beta, alpha]):
        raise ValueError('var, alpha, and/or beta must all be greater than 0.')
    if (any(i is not None for i in [mean, var]) and
            any(i is not None for i in [alpha, beta])):
        raise ValueError('Please specify either mean and var, '
                         'or alpha and beta')
    if (mean is not None and var is None) or \
            (mean is None and var is not None):
        raise ValueError('mean and var must be specified together')
    if (alpha is not None and beta is None) or \
            (alpha is None and beta is not None):
        raise ValueError('alpha and beta must be specified together')

    if mean is not None:
        alpha = mean**2/var
        beta = mean/var
    else:
        mean = alpha/beta
        var = alpha/beta**2

    x = np.linspace(max(mean - np.sqrt(var)*7, 0), mean + np.sqrt(var)*7, 1000)
    dist = stats.gamma(a=alpha, scale=1/beta)
    y = dist.pdf(x)
    sns.lineplot(x, y, color='green')
    plt.fill_between(x, y, color='green', alpha=.2)
    sns.despine(left=True)
    plt.yticks([], [])
    plt.title('Gamma({0}, {1}) Distribution PDF (mean and variance)'
              .format(mean, var),
              fontweight='bold', fontsize=14)


def lognormal_dist(mean, var):
    """Plot a log normal distribution.

    Pick either (mean and var) or (alpha and beta)
    Arguments:
        mean {int} -- mean parameter to log normal distribution.
            Must be greater than 0.
        var {int} -- var parameter to log normal distribution.
            Must be greater than 0.
    """
    if mean <= 0 or var <= 0:
        raise ValueError('mean and var must both be greater than 0.')

    mean = np.log(mean)
    var = np.log(var)
    x = np.linspace(max(mean - np.sqrt(var)*7, 0), mean + np.sqrt(var)*7, 1000)
    dist = stats.lognorm(np.sqrt(var), loc=mean)
    y = dist.pdf(x)
    sns.lineplot(x, y, color='green')
    plt.fill_between(x, y, color='green', alpha=.2)
    sns.despine(left=True)
    plt.yticks([], [])
    plt.title('Log-Normal({0:.2}, {1:.2}) Distribution PDF (mean and variance)'
              .format(mean, var),
              fontweight='bold', fontsize=14)


def poisson_dist(lam):
    """Plot a Poisson distribution.

    Arguments:
        lam {int} -- mean value per unit time. Must be greater than 0.
    """
    if lam <= 0:
        raise ValueError('Lambda must be greater than 0')

    x = np.linspace(max(lam - 4 * lam, 0), lam + 4 * lam, 1000)
    # dist = stats.poisson(lam)
    # y = dist.pmf(x, lam)
    y = ((lam**x)*np.exp(-lam))/factorial(x)
    sns.lineplot(x, y, color='green')
    plt.fill_between(x, y, color='green', alpha=.2)
    sns.despine(left=True)
    plt.yticks([], [])
    plt.title('Poisson({}) Distribution PMF'.format(lam),
              fontweight='bold', fontsize=14)
