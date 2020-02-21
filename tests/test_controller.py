"""run file for ab_test_model test functions."""

import model_tests as mt
import dist_explorer_tests as dt


def run_test():
    """Run a a single test function to verify basic run-ability."""
    # mt.two_variants_conversion()
    mt.one_variant_continuous()
    print('SUCCESS')


def run_basic_tests():
    """Run a subset of test_functions to test most of the functionality."""
    tests = ['one_variant_conversion',
             'one_variant_continuous',
             'two_variants_poisson']
    for test in tests:
        if '__' in test or len(test) <= 2:
            continue
        print('+ running test {}()'.format(test))
        getattr(mt, test)()
    print('SUCCESS')


def run_all():
    """Run every test function for ab test model."""
    for test in dir(mt):
        if '__' in test or len(test) <= 2:
            continue
        print('+ running test {}()'.format(test))
        getattr(mt, test)()
    print('SUCCESS')


def run_prior_tests():
    """Run every test function for prior plotting."""
    dt.test_beta()
    dt.test_gamma()
    dt.test_poisson()
    dt.test_lognormal()
    print('SUCCESS')


if __name__ == '__main__':
    run_test()
    # run_basic_tests()
    # run_all()

    # run_prior_tests()
