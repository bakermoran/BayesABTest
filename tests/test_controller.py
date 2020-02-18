"""run file for ab_test_model test functions."""

import tests as t


def run_test():
    """Run a subset of the test functions."""
    t.two_variants_poisson()


def run_basic_tests():
    """Run a subset of test_functions to test most of the functionality."""
    tests = ['one_variant_conversion',
             'one_variant_continuous',
             'two_variants_poisson']
    for test in tests:
        if '__' in test or len(test) <= 2:
            continue
        print('+ running test {}()'.format(test))
        getattr(t, test)()
        print('SUCCESS')


def run_all():
    """Run every test function."""
    for test in dir(t):
        if '__' in test or len(test) <= 2:
            continue
        print('+ running test {}()'.format(test))
        getattr(t, test)()
        print('SUCCESS')


if __name__ == '__main__':
    run_basic_tests()
    # run_test()
    # run_all()
