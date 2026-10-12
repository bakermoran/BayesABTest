"""Pure conjugate-update and beta comparison math.

Nothing in here touches dataframes or models, so every function can be
unit tested directly against closed forms.
"""

import numpy as np
import scipy.special as sc
from scipy import integrate, stats


def beta_posterior(prior_alpha, prior_beta, successes, trials):
    """Beta-Binomial update. Returns the posterior (alpha, beta)."""
    return prior_alpha + successes, prior_beta + trials - successes


def gamma_poisson_posterior(prior_alpha, prior_beta, total, n):
    """Gamma-Poisson update for n counts summing to total.

    Returns the posterior (alpha, rate).
    """
    return prior_alpha + total, prior_beta + n


def normal_posterior(prior_mean, prior_var, values):
    """Normal-Normal update for the mean, treating the sample variance as known.

    Returns the posterior (mean, var) of the mean.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    variance = np.var(values)
    if variance == 0:
        raise ValueError('values must not all be identical')
    post_var = 1 / (1 / prior_var + n / variance)
    post_mean = post_var * (prior_mean / prior_var
                            + values.mean() * n / variance)
    return post_mean, post_var


def prob_beta_greater(a, b, c, d):
    """Probability that X1 ~ Beta(a, b) is greater than X2 ~ Beta(c, d).

    Uses the closed form sum when c is an integer, and numerical
    integration otherwise (e.g. when an empirical prior has been scaled).
    """
    if float(c).is_integer():
        j = np.arange(0, int(c))
        w = sc.betaln(a + j, b + d)
        x = np.log(d + j)
        y = sc.betaln(1 + j, d)
        z = sc.betaln(a, b)
        return 1 - np.exp(w - x - y - z).sum()
    # P(X1 > X2) = int f2(x) * (1 - F1(x)) dx. Only integrate where X2 has
    # mass, so a narrow posterior is not missed by the quadrature.
    mean = c / (c + d)
    sd = np.sqrt(c * d / ((c + d) ** 2 * (c + d + 1)))
    lo, hi = max(0.0, mean - 12 * sd), min(1.0, mean + 12 * sd)
    value, _ = integrate.quad(
        lambda x: stats.beta.pdf(x, c, d) * stats.beta.sf(x, a, b),
        lo, hi, points=[mean], limit=200)
    return value


def expected_loss_beta(a, b, c, d, loss_type):
    """Compute the loss of choosing X2 ~ Beta(c, d) when X1 ~ Beta(a, b) is bigger.

    'absolute' is E[max(X1 - X2, 0)]. 'percent' is E[max(X1 - X2, 0) / X2],
    the loss in lift terms.
    """
    if loss_type == 'absolute':
        e1 = a / (a + b)
        e2 = c / (c + d)
        return (e1 * prob_beta_greater(a + 1, b, c, d)
                - e2 * prob_beta_greater(a, b, c + 1, d))
    if loss_type == 'percent':
        if c <= 1:
            raise ValueError('percent loss requires the alpha of the '
                             'chosen variant to be greater than 1')
        e1 = a / (a + b)
        e_inv2 = (c + d - 1) / (c - 1)
        return (e1 * e_inv2 * prob_beta_greater(a + 1, b, c - 1, d)
                - prob_beta_greater(a, b, c, d))
    raise ValueError("loss_type must be either 'absolute' or 'percent'")
