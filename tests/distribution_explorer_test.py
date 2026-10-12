"""AB test distribution exploration functions tests."""

import matplotlib.pyplot as plt
import pytest
from scipy import stats

from BayesABTest import ab_test_dist_explorer as pe


@pytest.mark.parametrize('alpha, beta', [(1, 1), (20, 80), (100, 900),
                                         (2.5, 7.5)])
def test_beta(alpha, beta):
    """The plotter returns the matching frozen distribution."""
    dist = pe.beta_dist(alpha, beta)
    assert dist.mean() == pytest.approx(alpha / (alpha + beta))


@pytest.mark.parametrize('alpha, beta', [(-1, 0), (-1, -5), (0, 5)])
def test_beta_invalid(alpha, beta):
    with pytest.raises(ValueError):
        pe.beta_dist(alpha, beta)


@pytest.mark.parametrize('mu', [15, 50, 2, 2.5])
def test_poisson(mu):
    dist = pe.poisson_dist(mu)
    assert dist.mean() == mu


@pytest.mark.parametrize('mu', [0, -5])
def test_poisson_invalid(mu):
    with pytest.raises(ValueError):
        pe.poisson_dist(mu)


def test_poisson_plots_a_pmf_on_integers():
    fig, ax = plt.subplots()
    pe.poisson_dist(4, ax=ax)
    x, y = ax.lines[0].get_data()
    assert all(float(v).is_integer() for v in x)
    assert y.sum() == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize('mean, var', [(650, 1.2), (300, 2.0), (1000, 0.01),
                                       (5, 4), (2, 0.5)])
def test_lognormal_matches_requested_mean_and_var(mean, var):
    dist = pe.lognormal_dist(mean, var)
    assert dist.mean() == pytest.approx(mean)
    assert dist.var() == pytest.approx(var)


@pytest.mark.parametrize('mean, var', [(-1, 0), (-1, -5), (0, 5), (5, 0)])
def test_lognormal_invalid(mean, var):
    with pytest.raises(ValueError):
        pe.lognormal_dist(mean, var)


def test_gamma_both_parameterizations_agree():
    by_shape = pe.gamma_dist(alpha=8, beta=2)
    by_moments = pe.gamma_dist(mean=4, var=2)
    assert by_shape.mean() == pytest.approx(4)
    assert by_shape.var() == pytest.approx(2)
    assert by_moments.mean() == pytest.approx(4)
    assert by_moments.var() == pytest.approx(2)
    assert isinstance(by_shape, stats._distn_infrastructure.rv_frozen)


@pytest.mark.parametrize('kwargs', [
    {}, {'mean': 1}, {'alpha': 1}, {'mean': 1, 'var': 1, 'alpha': 1, 'beta': 1},
    {'mean': 0, 'var': 1}, {'alpha': -1, 'beta': 1}])
def test_gamma_invalid(kwargs):
    with pytest.raises(ValueError):
        pe.gamma_dist(**kwargs)
