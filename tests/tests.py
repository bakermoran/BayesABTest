"""ab_test_model test functions."""

from BayesABTest import ab_test_model as ab
import data_helpers as dh
from matplotlib import pyplot as plt


def one_variant_conversion():
    """Create data and run a one variant report for a conversion metric."""
    raw_data_auto_bind = dh.create_conversion_data([.27, .3], ['off', 'on'],
                                                   metric_name='bind')
    auto_bind = ab(raw_data_auto_bind, metric='bind',
                   prior_info='uninformed', prior_func='beta', debug=True,
                   samples=1000)
    auto_bind.fit()
    auto_bind.plot()
    plt.show()
    auto_bind.loss_function()


def one_variant_continuous():
    """Create data and run a one variant report for a continuous metric."""
    # test between off and on buckets (small difference)
    rawdata_small = dh.create_continuous_data([600, 610], [1.5, 1.5],
                                              ['off', 'on'],
                                              metric_name='total_premium')
    premium_small = ab(rawdata_small, metric='total_premium',
                       prior_info='informed', prior_func='log-normal',
                       debug=True, samples=1000)
    premium_small.fit()
    premium_small.plot()
    plt.show()

    # test between off and on buckets (big difference)
    rawdata_big = dh.create_continuous_data([600, 700], [1.5, 1.5],
                                            ['off', 'on'],
                                            metric_name='total_premium')
    premium_big = ab(rawdata_big, metric='total_premium',
                     prior_info='informed', prior_func='log-normal',
                     debug=True, samples=1000)
    premium_big.fit()
    premium_big.plot()
    plt.show()


def one_variant_continuous_small_mean():
    """Create data and run a one variant report for a continuous metric.

    With a mean between 0 and 1.
    """
    rawdata_small = dh.create_continuous_data([.65, .66],
                                              [1.5, 1.5],
                                              ['off', 'on'],
                                              metric_name='loss_ratio')
    premium_small = ab(rawdata_small, metric='loss_ratio',
                       prior_info='informed',
                       prior_func='log-normal',
                       debug=True,
                       samples=1000)
    premium_small.fit()
    premium_small.plot()
    plt.show()


def two_variants_conversion():
    """Create data and run a two variant report for a conversion metric."""
    raw_data_2vars = dh.create_conversion_data([.22, .23, .235],
                                               ['control',
                                                'variant_1',
                                                'variant_2'],
                                               metric_name='bind')
    auto_bind = ab(raw_data_2vars, metric='bind',
                   prior_info='uninformed', prior_func='beta',
                   debug=True, control_bucket_name='control',
                   variant_bucket_names=['variant_1', 'variant_2'],
                   compare_variants=True, samples=1000)
    auto_bind.fit()
    auto_bind.plot(lift_plot_flag=True)
    plt.show()


def conversion_negative_variants():
    """Create data and run a two variant report for a conversion metric.

    Where the variants are worse than the control.
    """
    raw_data_2vars = dh.create_conversion_data([.25, .23, .235],
                                               ['control',
                                                'variant_1',
                                                'variant_2'],
                                               metric_name='bind')
    auto_bind = ab(raw_data_2vars, metric='bind',
                   prior_info='uninformed', prior_func='beta',
                   debug=True, control_bucket_name='control',
                   variant_bucket_names=['variant_1', 'variant_2'],
                   compare_variants=True, samples=1000)
    auto_bind.fit()
    auto_bind.plot(lift_plot_flag=True)
    plt.show()


def two_variants_continuous():
    """Create data and run a two variant report for a continuous metric."""
    rawdata = dh.create_continuous_data([600, 610, 615], [1.5, 1.5, 1.5],
                                        ['control',
                                         'variant_1',
                                         'variant_2'],
                                        metric_name='total_premium')

    premium = ab(rawdata, metric='total_premium', prior_info='informed',
                 prior_func='log-normal', debug=True,
                 control_bucket_name='control',
                 variant_bucket_names=['variant_1', 'variant_2'],
                 compare_variants=True,
                 samples=1000)
    premium.fit()
    premium.plot()
    plt.show()


def three_variants_continuous():
    """Create data and run a three variant report for a continuous metric."""
    rawdata = dh.create_continuous_data([600, 610, 615, 620],
                                        [1.5, 1.5, 1.5, 1.5],
                                        ['control',
                                         'variant_1',
                                         'variant_2',
                                         'variant_3'],
                                        metric_name='total_premium')
    premium = ab(rawdata, metric='total_premium',
                 prior_info='informed', prior_func='log-normal',
                 debug=True, control_bucket_name='control',
                 variant_bucket_names=['variant_1', 'variant_2', 'variant_3'],
                 compare_variants=True,
                 samples=1000)
    premium.fit()
    premium.plot()
    plt.show()


def four_variants_continuous():
    """Create data and run a four variant report for a continuous metric."""
    rawdata = dh.create_continuous_data([600, 610, 615, 620, 625],
                                        [1.5, 1.5, 1.5, 1.5, 1.5],
                                        ['control',
                                         'variant_1',
                                         'variant_2',
                                         'variant_3',
                                         'variant_4'],
                                        metric_name='total_premium')
    premium = ab(rawdata, metric='total_premium',
                 prior_info='informed', prior_func='log-normal',
                 debug=True, control_bucket_name='control',
                 variant_bucket_names=['variant_1', 'variant_2',
                                       'variant_3', 'variant_4'],
                 compare_variants=False, samples=1000)
    premium.fit()
    premium.plot(lift_plot_flag=True)
    plt.show()


def two_variants_continuous_normal():
    """Create data and run a four variant report.

    For a normal continuous metric.
    """
    rawdata = dh.create_continuous_data([600, 601, 602],
                                        [30, 30, 30],
                                        ['control', 'variant_1', 'variant_2'],
                                        metric_name='total_premium', log=False)
    premium = ab(rawdata, metric='total_premium',
                 prior_info='informed', prior_func='normal',
                 debug=True, control_bucket_name='control',
                 variant_bucket_names=['variant_1', 'variant_2'],
                 compare_variants=True, samples=1000)
    premium.fit()
    premium.plot(lift_plot_flag=True)
    plt.show()


def two_variants_poisson():
    """Create data and run a four variant report for a poisson metric."""
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='informed', prior_func='poisson',
                debug=True, control_bucket_name='control',
                variant_bucket_names=['variant_1', 'variant_2'],
                compare_variants=True, samples=3000)
    visits.fit()
    visits.plot(lift_plot_flag=True)
    plt.show()


def single_plot_posteriors_unit_test():
    """Test plot_posteriors.

    Use a poisson metric to test the public version of plot_posteriors,
    with various subsets of the variants.
    """
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='informed', prior_func='poisson',
                debug=True, control_bucket_name='control',
                variant_bucket_names=['variant_1', 'variant_2'],
                compare_variants=True, samples=3000)
    visits.fit()
    visits.plot_posteriors()
    plt.show()
    visits.plot_posteriors(variants=['control', 'variant_1', 'variant_2'])
    plt.show()
    visits.plot_posteriors(variants=['variant_1', 'variant_2'])
    plt.show()
    visits.plot_posteriors(variants=['control', 'variant_2'])
    plt.show()
    visits.plot_posteriors(variants=['control', 'variant_1'])
    plt.show()
    visits.plot_posteriors(variants=['variant_1'])
    plt.show()
    visits.plot_posteriors(variants=['control'])
    plt.show()
    visits.plot_posteriors(variants=['variant_2'])
    plt.show()


def single_plot_lift_unit_test():
    """Test plot_positive_lift.

    Use a poisson metric to test the public version of plot_positive_lift,
    with various subsets of the variants.
    """
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='informed', prior_func='poisson',
                debug=True, control_bucket_name='control',
                variant_bucket_names=['variant_1', 'variant_2'],
                compare_variants=True, samples=3000)
    visits.fit()
    visits.plot_positive_lift('control', 'variant_1')
    plt.show()
    visits.plot_positive_lift('variant_1', 'variant_2')
    plt.show()
    visits.plot_positive_lift('control', 'variant_2')
    plt.show()

    visits.plot_positive_lift('variant_1', 'control')
    plt.show()
    visits.plot_positive_lift('variant_2', 'variant_1')
    plt.show()
    visits.plot_positive_lift('variant_2', 'control')
    plt.show()


def single_plot_ecdf_unit_test():
    """Test plot_ecdf.

    Use a poisson metric to test the public version of plot_ecdf,
    with various subsets of the variants.
    """
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='informed', prior_func='poisson',
                debug=True, control_bucket_name='control',
                variant_bucket_names=['variant_1', 'variant_2'],
                compare_variants=True, samples=3000)
    visits.fit()
    visits.plot_ecdf('control', 'variant_1')
    plt.show()
    visits.plot_ecdf('variant_1', 'variant_2')
    plt.show()
    visits.plot_ecdf('control', 'variant_2')
    plt.show()

    visits.plot_ecdf('variant_1', 'control')
    plt.show()
    visits.plot_ecdf('variant_2', 'variant_1')
    plt.show()
    visits.plot_ecdf('variant_2', 'control')
    plt.show()


def test_specified_prior():
    """Test a user specified prior for several different types."""
    raw_data_2vars = dh.create_conversion_data([.22, .23, .235],
                                               ['control',
                                                'variant_1',
                                                'variant_2'],
                                               metric_name='bind')
    prior = {'alpha': 22, 'beta': 100-22}
    auto_bind = ab(raw_data_2vars, metric='bind',
                   prior_info='specified', prior_func='beta',
                   debug=True, control_bucket_name='control',
                   variant_bucket_names=['variant_1', 'variant_2'],
                   compare_variants=True, prior_parameters=prior,
                   samples=1000)
    auto_bind.fit()
    auto_bind.plot(lift_plot_flag=True)
    plt.show()

    prior = {'mean': 650, 'var': 1.5}
    rawdata = dh.create_continuous_data([600, 610, 615],
                                        [1.5, 1.5, 1.5],
                                        ['control', 'variant_1', 'variant_2'],
                                        metric_name='total_premium')
    premium = ab(rawdata, metric='total_premium',
                 prior_info='specified', prior_func='log-normal',
                 debug=True, control_bucket_name='control',
                 variant_bucket_names=['variant_1', 'variant_2'],
                 compare_variants=True, prior_parameters=prior,
                 samples=1000)
    premium.fit()
    premium.plot()
    plt.show()

    prior = {'mean': 650, 'var': 30}
    rawdata = dh.create_continuous_data([600, 601, 602],
                                        [30, 30, 30],
                                        ['control', 'variant_1', 'variant_2'],
                                        metric_name='total_premium', log=False)
    premium = ab(rawdata, metric='total_premium',
                 prior_info='specified', prior_func='normal',
                 debug=True, control_bucket_name='control',
                 variant_bucket_names=['variant_1', 'variant_2'],
                 compare_variants=True, prior_parameters=prior,
                 samples=1000)
    premium.fit()
    premium.plot(lift_plot_flag=True)
    plt.show()

    prior = {'mean': 15, 'var': 3}
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='specified', prior_func='poisson',
                debug=True, control_bucket_name='control',
                variant_bucket_names=['variant_1', 'variant_2'],
                compare_variants=True, prior_parameters=prior,
                samples=3000)
    visits.fit()
    visits.plot(lift_plot_flag=True)
    plt.show()

    prior = {'alpha': 8, 'beta': 2}
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='specified', prior_func='poisson',
                debug=True, control_bucket_name='control',
                variant_bucket_names=['variant_1', 'variant_2'],
                compare_variants=True, prior_parameters=prior,
                samples=3000)
    visits.fit()
    visits.plot(lift_plot_flag=True)
    plt.show()
