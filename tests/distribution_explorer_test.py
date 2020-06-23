"""AB test distribution exploration functions tests."""

from BayesABTest import ab_test_dist_explorer as pe
import pytest


@pytest.mark.parametrize("""alpha, beta""",
                         [(1, 1),
                          (20, 80),
                          (100, 900),
                          (-1, 0),
                          (-1, -5),
                          (0, 5)])
def test_beta(alpha, beta):
    """Test the beta plotter with various params."""
    if alpha > 0 and beta > 0:
        pe.beta_dist(alpha, beta)
    else:
        try:
            pe.beta_dist(alpha, beta)
        except ValueError:
            pass


@pytest.mark.parametrize("""mu""",
                         [(15), (50), (2), (0), (-5)])
def test_poisson(mu):
    """Test the poisson plotter with various params."""
    if mu > 0 and isinstance(mu, int):
        pe.poisson_dist(mu)
    else:
        try:
            pe.poisson_dist(mu)
        except ValueError:
            pass


@pytest.mark.parametrize("""mean, var""",
                         [(650, 1.2),
                          (300, 2.0),
                          (1000, 0.01),
                          (-1, 0),
                          (-1, -5),
                          (0, 5)])
def test_lognormal(mean, var):
    """Test the lognormal plotter with various params."""
    if mean > 0 and var > 0:
        pe.lognormal_dist(mean, var)
    else:
        try:
            pe.lognormal_dist(mean, var)
        except ValueError:
            pass


def test_gamma():
    """Test the gamma plotter with various params."""
    pe.gamma_dist(alpha=8, beta=2)
    pe.gamma_dist(mean=100, var=10)

    try:
        pe.gamma_dist(mean=100, beta=900)
    except ValueError:
        pass

    try:
        pe.gamma_dist(alpha=500, var=80)
    except ValueError:
        pass
