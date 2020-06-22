"""ab_test_model test functions."""

from BayesABTest import ab_test_model as ab
import data_helpers as dh
import pytest


@pytest.mark.parametrize("""control_rate, variant_rate,
                          control_name, variant_name""",
                         [(.27, .3, 'off', 'on'),
                          (.27, .23, 'control', 'variant'),
                          (.5, .35, 'old_brand', 'new_brand')])
def test_one_variant_conversion(control_rate, variant_rate,
                                control_name, variant_name):
    """Create data and run a one variant report for a conversion metric."""
    raw_data_auto_bind = dh.create_conversion_data([control_rate,
                                                    variant_rate],
                                                   [control_name,
                                                    variant_name],
                                                   metric_name='bind')
    auto_bind = ab(raw_data_auto_bind, metric='bind',
                   control_bucket_name=control_name,
                   prior_info='uninformed', prior_func='beta', debug=True,
                   samples=1000)
    auto_bind.fit()
    auto_bind.plot()
