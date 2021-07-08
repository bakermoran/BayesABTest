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
                   prior_info='uninformed', prior_function='beta', debug=True,
                   samples=1000)
    auto_bind.fit()
    auto_bind.plot()


@pytest.mark.parametrize("""control_rate, variant_rate,
                          control_variance, variant_variance,
                          control_name, variant_name""",
                         [(600, 630, 1.5, 1.5, 'off', 'on'),
                          (600, 700, 1.5, 1.5, 'control', 'variant'),
                          (.85, .65, 1.5, 1.5, 'old_price', 'mcmodel')])
def test_one_variant_continuous(control_rate, variant_rate,
                                control_variance, variant_variance,
                                control_name, variant_name):
    """Create data and run a one variant report for a continuous metric."""
    raw_data_premium = dh.create_continuous_data([control_rate,
                                                  variant_rate],
                                                 [control_variance,
                                                  variant_variance],
                                                 [control_name,
                                                  variant_name],
                                                 metric_name='total_premium')
    premium_test = ab(raw_data_premium, metric='total_premium',
                      control_bucket_name=control_name,
                      prior_info='informed', prior_function='log-normal',
                      debug=True, samples=1000)
    premium_test.fit()
    premium_test.plot()


@pytest.mark.parametrize("""control_rate, variant1_rate, variant2_rate,
                          control_name, variant1_name, variant2_name""",
                         [(.2, .3, .4,  'control', 'variant1', 'variant2'),
                          (.25, .23, .235, 'control', 'variant1', 'variant2')])
def test_two_variant_conversion(control_rate, variant1_rate, variant2_rate,
                                control_name, variant1_name, variant2_name):
    """Create data and run a two variant report for a conversion metric."""
    raw_data_auto_bind = dh.create_conversion_data([control_rate,
                                                    variant1_rate,
                                                    variant2_rate],
                                                   [control_name,
                                                    variant1_name,
                                                    variant2_name],
                                                   metric_name='conversion')
    auto_bind = ab(raw_data_auto_bind, metric='conversion',
                   control_bucket_name=control_name, compare_variants=True,
                   prior_info='uninformed', prior_function='beta', debug=True,
                   samples=1000)
    auto_bind.fit()
    auto_bind.plot()


@pytest.mark.parametrize("""control_rate, variant1_rate, variant2_rate,
                          control_variance, variant1_variance,
                          variant2_variance, control_name, variant1_name,
                          variant2_name""",
                         [(600, 610, 615, 1.5, 1.5, 1.5, 'control', 'variant1',
                           'variant2'),
                          (.85, .65, .75, 1.5, 1.5, 1.5, 'old_price',
                           'mcmodel1', 'mcmodel2')])
def test_two_variant_continuous(control_rate, variant1_rate, variant2_rate,
                                control_variance, variant1_variance,
                                variant2_variance, control_name, variant1_name,
                                variant2_name):
    """Create data and run a two variant report for a continuous metric."""
    raw_data_premium = dh.create_continuous_data([control_rate,
                                                  variant1_rate,
                                                  variant2_rate],
                                                 [control_variance,
                                                  variant1_variance,
                                                  variant2_variance],
                                                 [control_name,
                                                  variant1_name,
                                                  variant2_name],
                                                 metric_name='total_premium')
    premium_test = ab(raw_data_premium, metric='total_premium',
                      control_bucket_name=control_name,
                      prior_info='informed', prior_function='log-normal',
                      debug=True, samples=1000)
    premium_test.fit()
    premium_test.plot()


@pytest.mark.parametrize("""control_rate, variant1_rate, variant2_rate,
                          control_variance, variant1_variance,
                          variant2_variance, control_name, variant1_name,
                          variant2_name, log""",
                         [(600, 610, 615, 1.5, 1.5, 1.5, 'control', 'variant1',
                           'variant2', True),
                          (600, 601, 602, 30, 30, 30, 'old_price', 'mcmodel1',
                           'mcmodel2', False)])
def test_gamma_uninformed(control_rate, variant1_rate, variant2_rate,
                          control_variance, variant1_variance,
                          variant2_variance, control_name, variant1_name,
                          variant2_name, log):
    """Create data and run a two variant report for a continuous metric.

       While testing the uninformed prior for a gamma distribution.
       """
    rawdata = dh.create_continuous_data([control_rate,
                                         variant1_rate,
                                         variant2_rate],
                                        [control_variance,
                                         variant1_variance,
                                         variant2_variance],
                                        [control_name,
                                         variant1_name,
                                         variant2_name],
                                        metric_name='total_premium',
                                        log=log)
    prior = 'log-normal' if log else 'normal'
    premium = ab(rawdata, metric='total_premium', prior_info='uninformed',
                 prior_function=prior, debug=True, compare_variants=True,
                 control_bucket_name=control_name, samples=1000)
    premium.fit()
    premium.plot()


@pytest.mark.parametrize("""control_rate, variant1_rate, variant2_rate,
                          control_name, variant1_name, variant2_name""",
                         [(10, 20, 25,  'control', 'variant1', 'variant2'),
                          (25, 50, 25, 'control', 'variant1', 'variant2')])
def test_two_variants_poisson(control_rate, variant1_rate, variant2_rate,
                              control_name, variant1_name, variant2_name):
    """Create data and run a two variant report for a poisson metric."""
    rawdata = dh.create_poisson_data([control_rate, variant1_rate,
                                      variant2_rate],
                                     [control_name, variant1_name,
                                      variant2_name],
                                     metric_name='visits', sample_length=3000)
    rawdata = rawdata.sample(frac=1).reset_index(drop=True)
    visits = ab(rawdata, metric='visits',
                prior_info='informed', prior_function='poisson',
                debug=True, control_bucket_name=control_name,
                compare_variants=True, samples=3000)
    visits.fit()
    visits.plot()


def test_three_variants_continuous():
    """Create data and run a three variant report for a continuous metric."""
    rawdata = dh.create_continuous_data([600, 610, 615, 620],
                                        [1.5, 1.5, 1.5, 1.5],
                                        ['control', 'variant_1', 'variant_2',
                                         'variant_3'],
                                        metric_name='total_premium')
    premium = ab(rawdata, metric='total_premium',
                 prior_info='informed', prior_function='log-normal',
                 debug=True, control_bucket_name='control',
                 compare_variants=True, samples=1000)
    premium.fit()
    premium.plot()


def test_four_variants_continuous():
    """Create data and run a four variant report for a continuous metric."""
    rawdata = dh.create_continuous_data([600, 610, 615, 620, 625],
                                        [1.1, 1.1, 1.1, 1.1, 1.1],
                                        ['control', 'variant_1',
                                         'variant_2', 'variant_3',
                                         'variant_4'],
                                        metric_name='total_premium',
                                        sample_length=1000)
    premium = ab(rawdata, metric='total_premium',
                 prior_info='informed', prior_function='log-normal',
                 debug=True, control_bucket_name='control',
                 compare_variants=False, samples=1000)
    premium.fit()
    premium.plot(lift_plot_flag=True)


def test_specified_prior_conversion():
    """E2E test a user specified prior for conversion."""
    raw_data_2vars = dh.create_conversion_data([.22, .23, .235],
                                               ['control',
                                                'variant_1',
                                                'variant_2'],
                                               metric_name='conversion')
    prior = {'alpha': 22, 'beta': 100-22}
    site_conversion = ab(raw_data_2vars, metric='conversion',
                         prior_info='specified', prior_function='beta',
                         debug=True, control_bucket_name='control',
                         compare_variants=True, prior_parameters=prior,
                         samples=1000)
    site_conversion.fit()
    site_conversion.plot(lift_plot_flag=True)


def test_specified_prior_continuous():
    """E2E test a user specified prior for continuous."""
    prior = {'mean': 650, 'var': 1.5}
    rawdata = dh.create_continuous_data([600, 610, 615],
                                        [1.5, 1.5, 1.5],
                                        ['control', 'variant_1', 'variant_2'],
                                        metric_name='total_premium')
    premium = ab(rawdata, metric='total_premium',
                 prior_info='specified', prior_function='log-normal',
                 debug=True, control_bucket_name='control',
                 compare_variants=True, prior_parameters=prior,
                 samples=1000)
    premium.fit()
    premium.plot()


def test_specified_prior_normal():
    """E2E test a user specified prior for continuous, log=False."""
    prior = {'mean': 650, 'var': 30000}
    rawdata = dh.create_continuous_data([600, 601, 602],
                                        [30, 30, 30],
                                        ['control', 'variant_1', 'variant_2'],
                                        metric_name='total_premium', log=False)
    premium = ab(rawdata, metric='total_premium',
                 prior_info='specified', prior_function='normal',
                 debug=True, control_bucket_name='control',
                 compare_variants=True, prior_parameters=prior,
                 samples=1000)
    premium.fit()
    premium.plot(lift_plot_flag=True)


def test_specified_prior_poisson():
    """E2E test a user specified prior for poisson."""
    prior = {'mean': 15, 'var': 3}
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='specified', prior_function='poisson',
                debug=True, control_bucket_name='control',
                compare_variants=True, prior_parameters=prior,
                samples=3000)
    visits.fit()
    visits.plot(lift_plot_flag=True)


def test_specified_prior_poisson_alpha_beta():
    """E2E test a user specified prior for a beta distribution."""
    prior = {'alpha': 8, 'beta': 2}
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='specified', prior_function='poisson',
                debug=True, control_bucket_name='control',
                compare_variants=True, prior_parameters=prior,
                samples=3000)
    visits.fit()
    visits.plot(lift_plot_flag=True)
