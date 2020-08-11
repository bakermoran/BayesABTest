"""Loss function functionality for ab_test_model."""

import numpy as np
import scipy.special as sc
import ab_test_dist_explorer as de
from matplotlib import pyplot as plt
# pylint: disable=no-member


def probability_B_beats_A(a_A, b_A, a_B, b_B):
    """Calculate the probablty that a Beta distributions is greater.

    Arguments:
        a_A {int} -- Beta_A alpha param
        b_A {int} -- Beta_A beta param
        a_B {int} -- Beta_B alpha param
        b_B {int} -- Beta_B beta param

    Returns:
        float -- probabilty that Beta_B(a_B, b_B) is greater
          than Beta_A(a_A, b_A)
    """
    sum = 0
    for i in range(0, a_B-1):
        sum += np.exp(sc.betaln(a_A+i, b_B+b_A) -
                      np.log(b_B+i) -
                      sc.betaln(1+i, b_B) -
                      sc.betaln(a_A, b_A)
                      )
    return sum


def probability_C_beats_A_and_B(a_A, b_A, a_B, b_B, a_C, b_C):
    """Calculate the probablty that a Beta distributions is greater than two.

    Arguments:
        a_A {int} -- Beta_A alpha param
        b_A {int} -- Beta_A beta param
        a_B {int} -- Beta_B alpha param
        b_B {int} -- Beta_B beta param
        a_C {int} -- Beta_C alpha param
        b_C {int} -- Beta_C beta param

    Returns:
        float -- probabilty that Beta_C(a_C, b_C) is greater
          than Beta_A(a_A, b_A) and Beta_A(a_B, b_B)
    """
    total = 0
    for i in range(0, a_A-1):
        for j in range(0, a_B-1):
            total += np.exp(sc.betaln(a_C+i+j, b_A+b_B+b_C) -
                            np.log(b_A+i) - np.log(b_B+j) -
                            sc.betaln(1+i, b_A) - sc.betaln(1+j, b_B) -
                            sc.betaln(a_C, b_C))
    return (1 - probability_B_beats_A(a_C, b_C, a_A, b_A)
              - probability_B_beats_A(a_C, b_C, a_B, b_B) + total)


def _loss_choose_B_over_A(a_A, b_A, a_B, b_B):
    """Calc loss function value of two dists Beta_A(a_A, b_B) and Beta_B(a_B, b_B).

    Arguments:
        a_A {int} -- Beta_A alpha param
        b_A {int} -- Beta_A beta param
        a_B {int} -- Beta_B alpha param
        b_B {int} -- Beta_B beta param

    Returns:
        float -- the loss function value for Beta_B over Beta_A
    """
    x1 = sc.betaln(a_A + 1, b_A)
    y1 = np.log(1 - probability_B_beats_A(a_A + 1, b_A, a_B, b_B))
    z1 = sc.betaln(a_A, b_A)

    x2 = sc.betaln(a_B + 1, b_B)
    y2 = np.log(1 - probability_B_beats_A(a_A, b_A, a_B + 1, b_B))
    z2 = sc.betaln(a_B, b_B)

    return np.exp(x1 + y1 - z1) - np.exp(x2 + y2 - z2)


def sample_size_fixed_probability(baseline_conversion_rate,
                                  expected_relative_lift, epsilon=0.0001,
                                  minimum_confidence=0.95,
                                  prior_alpha=1,
                                  prior_beta=1):

    if expected_relative_lift <= 0:
        raise ValueError('expected_relative_lift must be greater than 0')

    variant_conversion_rate = baseline_conversion_rate * (
                               1 + expected_relative_lift)

    confidence = 0
    base_sample = 10

    while confidence < minimum_confidence:
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

        confidence = probability_B_beats_A(a, b, c, d)
        base_sample += 10

    print('loss function: ', _loss_choose_B_over_A(a, b, c, d))
    print('probablility variant is truly better: ', confidence)
    print('required sample size per variant {0} (in total: {1})'
          .format(base_sample, base_sample*2))

    de.beta_dist(a, b)
    de.beta_dist(c, d)
    plt.show()


def sample_size_fixed_loss_tolerance(baseline_conversion_rate,
                                     expected_relative_lift,
                                     loss_tolerance=0.05,
                                     prior_alpha=1,
                                     prior_beta=1):

    if expected_relative_lift <= 0:
        raise ValueError('expected_relative_lift must be greater than 0')

    variant_conversion_rate = baseline_conversion_rate * (
                               1 + expected_relative_lift)

    epsilon = baseline_conversion_rate - baseline_conversion_rate * (
              1 - loss_tolerance)

    B_risk = float('inf')
    base_sample = 10

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

        B_risk = _loss_choose_B_over_A(a, b, c, d)
        confidence = probability_B_beats_A(a, b, c, d)
        base_sample += 10

    print('loss function: ', _loss_choose_B_over_A(a, b, c, d))
    print('probablility variant is truly better: ', confidence)
    print('required sample size per variant {0} (in total: {1})'
          .format(base_sample, base_sample*2))

    de.beta_dist(a, b)
    de.beta_dist(c, d)
    plt.show()


# sample_size_fixed_probability(baseline_conversion_rate=.11,
#                               expected_relative_lift=.1,
#                               minimum_confidence=0.95)

sample_size_fixed_loss_tolerance(baseline_conversion_rate=.11,
                                 expected_relative_lift=.1,
                                 loss_tolerance=.01)
