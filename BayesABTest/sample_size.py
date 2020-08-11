"""Loss function functionality for ab_test_model."""

import numpy as np
import scipy.special as sc
# pylint: disable=no-member


def _h(a, b, c, d):
    """Calculate the probablty that a Beta distributions is greater.

    Returns:
        float -- probabilty that Beta(a,b) is greater than Beta(c,d)
    """
    sum = 0
    for j in range(0, c):
        w = sc.betaln(a + j, b + d)
        x = np.log(d + j)
        y = sc.betaln(1 + j, d)
        z = sc.betaln(a, b)
        sum += np.exp(w - x - y - z)
    return 1 - sum


def _loss_beta(a, b, c, d, loss_type):
    """Calc loss function value of two dists Beta_1(a,b) and Beta_2(c,d).

    Arguments:
        a {int} -- Beta_1 alpha param
        b {int} -- Beta_1 beta param
        c {int} -- Beta_2 alpha param
        d {int} -- Beta_2 beta param
        loss_type {string} -- 'absolute' or 'percent', is the type of loss
            value returned.

    Returns:
        float -- the loss function value for Beta_1 over Beta_2
    """
    if loss_type == 'absolute':
        x1 = sc.betaln(a + 1, b)
        y1 = np.log(_h(a + 1, b, c, d))
        z1 = sc.betaln(a, b)

        x2 = sc.betaln(c + 1, d)
        y2 = np.log(_h(a, b, c + 1, d))
        z2 = sc.betaln(c, d)

        return np.exp(x1 + y1 - z1) - np.exp(x2 + y2 - z2)

    elif loss_type == 'percent':
        prob_1 = _h(a, b, c, d)

        x = sc.betaln(a - 1, b)
        y = sc.betaln(a, b)
        z = sc.betaln(c + 1, d)
        w = sc.betaln(c, d)
        prob_2 = np.log(_h(a - 1, b, c + 1, d))

        return prob_1 - np.exp(x - y + z - w + prob_2)


def sample_size(baseline_conversion_rate, expected_relative_lift,
                loss_type='absolute', epsilon=0.0001,
                prior_alpha=1, prior_beta=1):

    variant_conversion_rate = baseline_conversion_rate * \
                                (1 + expected_relative_lift)

    # A_risk = float('inf')
    B_risk = float('inf')
    base_sample = 10

    # while (A_risk > epsilon and B_risk > epsilon):
    while B_risk > epsilon:
        control_alpha_likelihood = int(round(base_sample *
                                             baseline_conversion_rate, 0))
        control_beta_likelihood = int(round(base_sample *
                                            (1 - baseline_conversion_rate), 0))
        variant_alpha_likelihood = int(round(base_sample *
                                             variant_conversion_rate, 0))
        variant_beta_likelihood = int(round(base_sample *
                                            (1 - variant_conversion_rate), 0))
        a = control_alpha_likelihood + prior_alpha
        b = control_beta_likelihood + prior_beta
        c = variant_alpha_likelihood + prior_alpha
        d = variant_beta_likelihood + prior_beta

        # A_risk = _loss_beta(c, d, a, b, loss_type=loss_type)
        B_risk = _loss_beta(a, b, c, d, loss_type=loss_type)
        base_sample += 10

    print('required sample size per variant {0} (in total: {1})'
          .format(base_sample, base_sample*2))


sample_size(baseline_conversion_rate=.30,
            expected_relative_lift=.2,
            epsilon=0.01)
