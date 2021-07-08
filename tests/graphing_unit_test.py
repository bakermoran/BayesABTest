"""unit testing graphing functions in ab_test_model."""

from BayesABTest import ab_test_model as ab
import data_helpers as dh


def test_single_plot_posteriors_unit():
    """Test plot_posteriors.

    Use a poisson metric to test the public version of plot_posteriors,
    with various subsets of the variants.
    """
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='informed', prior_function='poisson',
                debug=True, control_bucket_name='control',
                compare_variants=True, samples=3000)
    visits.fit()
    visits.plot_posteriors()
    visits.plot_posteriors(variants=['control', 'variant_1', 'variant_2'])
    visits.plot_posteriors(variants=['variant_1', 'variant_2'])
    visits.plot_posteriors(variants=['control', 'variant_2'])
    visits.plot_posteriors(variants=['control', 'variant_1'])
    visits.plot_posteriors(variants=['variant_1'])
    visits.plot_posteriors(variants=['control'])
    visits.plot_posteriors(variants=['variant_2'])
    print('PASSED: test_single_plot_posteriors_unit')


def test_single_plot_lift_unit():
    """Test plot_positive_lift.

    Use a poisson metric to test the public version of plot_positive_lift,
    with various subsets of the variants.
    """
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='informed', prior_function='poisson',
                debug=True, control_bucket_name='control',
                compare_variants=True, samples=3000)
    visits.fit()
    visits.plot_positive_lift('control', 'variant_1')
    visits.plot_positive_lift('variant_1', 'variant_2')
    visits.plot_positive_lift('control', 'variant_2')

    visits.plot_positive_lift('variant_1', 'control')
    visits.plot_positive_lift('variant_2', 'variant_1')
    visits.plot_positive_lift('variant_2', 'control')
    print('PASSED: test_single_plot_lift_unit')


def test_single_plot_ecdf_unit():
    """Test plot_ecdf.

    Use a poisson metric to test the public version of plot_ecdf,
    with various subsets of the variants.
    """
    rawdata = dh.create_poisson_data([15, 17, 20],
                                     ['control', 'variant_1', 'variant_2'],
                                     metric_name='visits', sample_length=3000)

    visits = ab(rawdata, metric='visits',
                prior_info='informed', prior_function='poisson',
                debug=True, control_bucket_name='control',
                compare_variants=True, samples=3000)
    visits.fit()
    visits.plot_ecdf('control', 'variant_1')
    visits.plot_ecdf('variant_1', 'variant_2')
    visits.plot_ecdf('control', 'variant_2')
    visits.plot_ecdf('variant_1', 'control')
    visits.plot_ecdf('variant_2', 'variant_1')
    visits.plot_ecdf('variant_2', 'control')
    print('PASSED: test_single_plot_ecdf_unit')
