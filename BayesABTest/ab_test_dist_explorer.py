"""Functions to help explore distribution parameters.

Each function plots the distribution on ``ax`` (default the current Axes) and
returns the frozen ``scipy.stats`` distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy import stats

__all__ = ['beta_dist', 'gamma_dist', 'lognormal_dist', 'poisson_dist']


def _draw(ax, x, y, title):
    ax = ax if ax is not None else plt.gca()
    ax.plot(x, y, color='green')
    ax.fill_between(x, y, color='green', alpha=.2)
    sns.despine(ax=ax, left=True)
    ax.set_yticks([])
    ax.set_title(title, fontweight='bold', fontsize=14)
    return ax


def beta_dist(alpha, beta, *, ax=None):
    """Plot a Beta distribution.

    Arguments:
        alpha {float} -- alpha parameter to beta function. Represents the
            amount of conversions. must be greater than 0.
        beta {float} -- beta parameter to beta function. Represents the amount
            of non-conversions. must be greater than 0.
        ax -- matplotlib Axes to draw on (default current Axes).
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError('alpha and beta must both be greater than 0.')

    dist = stats.beta(alpha, beta)
    x = np.linspace(0, 1, 1000)
    ax = _draw(ax, x, dist.pdf(x),
               'Beta({0}, {1}) Distribution PDF'.format(alpha, beta))
    ax.set_xlabel('Conversion Rate')
    ax.set_xlim((0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=None))
    return dist


def gamma_dist(mean=None, var=None, alpha=None, beta=None, *, ax=None):
    """Plot a Gamma distribution.

    Pick either (mean and var) or (alpha and beta), where beta is the rate.

    Arguments:
        mean {float} -- mean of the gamma distribution. Must be greater than 0.
        var {float} -- variance of the gamma distribution.
            Must be greater than 0.
        alpha {float} -- alpha parameter to gamma distribution.
            Must be greater than 0.
        beta {float} -- beta (rate) parameter to gamma distribution.
            Must be greater than 0.
        ax -- matplotlib Axes to draw on (default current Axes).
    """
    if any(i is not None and i <= 0 for i in [mean, var, beta, alpha]):
        raise ValueError('mean, var, alpha, and/or beta must all be greater '
                         'than 0.')
    if (any(i is not None for i in [mean, var]) and
            any(i is not None for i in [alpha, beta])):
        raise ValueError('Please specify either mean and var, '
                         'or alpha and beta')
    if (mean is None) != (var is None):
        raise ValueError('mean and var must be specified together')
    if (alpha is None) != (beta is None):
        raise ValueError('alpha and beta must be specified together')
    if mean is None and alpha is None:
        raise ValueError('Please specify either mean and var, '
                         'or alpha and beta')

    if mean is not None:
        alpha = mean**2/var
        beta = mean/var
    else:
        mean = alpha/beta
        var = alpha/beta**2

    dist = stats.gamma(a=alpha, scale=1/beta)
    x = np.linspace(max(mean - np.sqrt(var)*7, 0), mean + np.sqrt(var)*7, 1000)
    _draw(ax, x, dist.pdf(x),
          'Gamma(alpha={0:.3g}, beta={1:.3g}) Distribution PDF'
          .format(alpha, beta))
    return dist


def lognormal_dist(mean, var, *, ax=None):
    """Plot a log normal distribution from its mean and variance.

    Arguments:
        mean {float} -- mean of the log normal distribution.
            Must be greater than 0.
        var {float} -- variance of the log normal distribution.
            Must be greater than 0.
        ax -- matplotlib Axes to draw on (default current Axes).
    """
    if mean <= 0 or var <= 0:
        raise ValueError('mean and var must both be greater than 0.')

    # moment match the log-normal mean and var to the underlying normal
    sigma2 = np.log(1 + var / mean**2)
    mu = np.log(mean) - sigma2 / 2
    dist = stats.lognorm(np.sqrt(sigma2), scale=np.exp(mu))
    x = np.linspace(0, dist.ppf(0.999), 1000)
    _draw(ax, x, dist.pdf(x),
          'Log-Normal(mean={0:.3g}, var={1:.3g}) Distribution PDF'
          .format(mean, var))
    return dist


def poisson_dist(lam, *, ax=None):
    """Plot a Poisson distribution.

    Arguments:
        lam {float} -- mean value per unit time. Must be greater than 0.
        ax -- matplotlib Axes to draw on (default current Axes).
    """
    if lam <= 0:
        raise ValueError('Lambda must be greater than 0')

    dist = stats.poisson(lam)
    x = np.arange(0, int(np.ceil(lam + 5 * np.sqrt(lam) + 5)))
    _draw(ax, x, dist.pmf(x), 'Poisson({}) Distribution PMF'.format(lam))
    return dist
