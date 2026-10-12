"""Shared pytest fixtures."""

import os

# Headless by default so tests never open plot windows. Override by setting
# MPLBACKEND yourself (e.g. MPLBACKEND=macosx) before running pytest.
os.environ.setdefault('MPLBACKEND', 'Agg')

import data_helpers as dh  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from BayesABTest import ABTest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure after each test."""
    yield
    plt.close('all')


@pytest.fixture
def fitted_poisson_result():
    """Return a fitted three-bucket poisson Result."""
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)
    return ABTest('visits', 'poisson', control='control',
                  n_samples=3000, random_state=0).fit(rawdata)
