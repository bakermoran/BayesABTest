"""Shared pytest fixtures."""

import os

# Headless by default so tests never open plot windows. Override by setting
# MPLBACKEND yourself (e.g. MPLBACKEND=macosx) before running pytest.
os.environ.setdefault('MPLBACKEND', 'Agg')

import data_helpers as dh  # noqa: E402
import pytest  # noqa: E402

from BayesABTest import ab_test_model as ab  # noqa: E402


@pytest.fixture
def fitted_poisson_model():
    """Return a fitted three-bucket poisson ab_test_model."""
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)
    model = ab(rawdata, metric='visits',
               prior_info='informed', prior_function='poisson',
               control_bucket_name='control',
               compare_variants=True, samples=3000)
    model.fit()
    return model
