"""AB test distribution exploration functions tests."""

from BayesABTest import ab_test_dist_explorer as pe
from matplotlib import pyplot as plt


def tst_beta():
    """Test the beta plotter with various params."""
    pe.beta_dist(1, 1)
    plt.show()

    pe.beta_dist(20, 80)
    plt.show()

    pe.beta_dist(100, 900)
    plt.show()

    pe.beta_dist(500, 80)
    plt.show()

    try:
        pe.beta_dist(-1, 0)
        plt.show()
    except ValueError:
        print('Value error successfully caught')

    try:
        pe.beta_dist(-1, -5)
        plt.show()
    except ValueError:
        print('Value error successfully caught')

    try:
        pe.beta_dist(5, 0)
        plt.show()
    except ValueError:
        print('Value error successfully caught')


def tst_gamma():
    """Test the gamma plotter with various params."""
    pe.gamma_dist(alpha=8, beta=2)
    plt.show()

    pe.gamma_dist(mean=100, var=10)
    plt.show()

    try:
        pe.gamma_dist(mean=100, beta=900)
        plt.show()
    except ValueError:
        print('Value error successfully caught')

    try:
        pe.gamma_dist(alpha=500, var=80)
        plt.show()
    except ValueError:
        print('Value error successfully caught')


def tst_poisson():
    """Test the poisson plotter with various params."""
    pe.poisson_dist(15)
    plt.show()

    pe.poisson_dist(50)
    plt.show()

    pe.poisson_dist(2)
    plt.show()

    try:
        pe.poisson_dist(0)
        plt.show()
    except ValueError:
        print('Value error successfully caught')

    try:
        pe.poisson_dist(-5)
        plt.show()
    except ValueError:
        print('Value error successfully caught')


def tst_lognormal():
    """Test the lognormal plotter with various params."""
    pe.lognormal_dist(650, 1.2)
    plt.show()

    pe.lognormal_dist(300, 2.0)
    plt.show()

    pe.lognormal_dist(1000, 0.01)
    plt.show()

    try:
        pe.lognormal_dist(-1000, 1)
        plt.show()
    except ValueError:
        print('Value error successfully caught')

    try:
        pe.lognormal_dist(0, 1)
        plt.show()
    except ValueError:
        print('Value error successfully caught')

    try:
        pe.lognormal_dist(100, -1)
        plt.show()
    except ValueError:
        print('Value error successfully caught')

    try:
        pe.lognormal_dist(10, 0)
        plt.show()
    except ValueError:
        print('Value error successfully caught')
