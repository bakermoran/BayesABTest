"""run file for ab_test_model test functions."""

import legacy_model_tsts as mt
import legacy_dist_explorer_tsts as dt


def run_tst():
    """Run a a single test function to verify basic run-ability."""
    # mt.two_variants_conversion()
    mt.one_variant_continuous()
    print('SUCCESS')


def run_basic_tsts():
    """Run a subset of test_functions to test most of the functionality."""
    tsts = ['one_variant_conversion',
            'one_variant_continuous',
            'two_variants_poisson']
    for tst in tsts:
        if '__' in tst or len(tst) <= 2:
            continue
        print('+ running test {}()'.format(tst))
        getattr(mt, tst)()
    print('SUCCESS')


def run_all():
    """Run every test function for ab test model."""
    for tst in dir(mt):
        if '__' in tst or len(tst) <= 2:
            continue
        print('+ running test {}()'.format(tst))
        getattr(mt, tst)()
    print('SUCCESS')


def run_prior_tsts():
    """Run every test function for prior plotting."""
    dt.tst_beta()
    dt.tst_gamma()
    dt.tst_poisson()
    dt.tst_lognormal()
    print('SUCCESS')


if __name__ == '__main__':
    run_tst()
    # run_basic_tsts()
    # run_all()

    # run_prior_tsts()
