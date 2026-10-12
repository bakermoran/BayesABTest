"""Functions to help generate fake test data."""

import numpy as np
import pandas as pd


def _rng(rng):
    """Return rng, or a fresh unseeded Generator if None."""
    return rng if rng is not None else np.random.default_rng()


def create_conversion_sample(true_rate, length=1000, rng=None):
    """Create a conversion binary list of n=length samples with true_rate.

    Arguments:
        true_rate {float} - true conversion rate of the sample. on the
            interval [0,1].
        length {int} -- number of obersvations to be in the sample data
            (default: {1000})

    Returns:
        sample {list} -- list of conversion samples
    """
    return [int(x) for x in _rng(rng).random(length) < true_rate]


def create_continuous_sample(mean, sigma, log=True, length=1000, rng=None):
    """Create a random (log) normal list of n=length samples with mu/sigma.

    Arguments:
        Input values should not be logged.
        mean {float} -- mean value of the continuous distribution
        sigma {float} -- standard deviation of the continuous distribution
        log {bool} -- should this be a log-normal sample (default: {True})
        length {int} -- number of obersvations to be in the sample
            data (default: {1000})

    Returns:
        sample {list} -- list of continuous samples
    """
    if log:
        mean = np.log(mean)
        sigma = np.log(sigma)
        sample = list(_rng(rng).lognormal(mean=mean, sigma=sigma,
                                          size=length))
    else:
        sample = list(_rng(rng).normal(loc=mean, scale=sigma, size=length))
    return sample


def create_poisson_sample(mean, length=1000, rng=None):
    """Create a random poisson list of n=length samples with mean.

    Arguments:
        Input values should not be logged.
        mean {float} -- mean value of the poisson distribution
        length {int} -- number of obersvations to be in the sample
            data (default: {1000})

    Returns:
        sample {list} -- list of poisson samples
    """
    sample = list(_rng(rng).poisson(lam=mean, size=length))
    return sample


def create_conversion_data(true_rates, bucket_names, metric_name,
                           bucket_column='bucket', sample_length=1000, seed=0):
    """Create a dataframe to simulate conversion data.

    true_rates and bucket_names should be of length = number of variants.

    Arguments:
        true_rates {list} -- list of the true rates of the variants
        bucket_names {list} -- list strings of the bucket names of the
            variants
        metric_name {str} -- string for the name of the conversion metric
        bucket_column {str} -- string for the name of the column for the
            buckets
        sample_length {int} number of obersvations to be in each sample
            data (default: {1000})
        seed {int} -- seed for the data. One generator is shared by all the
            buckets (default: {0})

    Returns:
        raw_data {dataframe} -- dataframe
    """
    rng = np.random.default_rng(seed)
    data = {}
    metric = []
    buckets = []
    for rate, name in zip(true_rates, bucket_names):
        metric = metric + create_conversion_sample(rate, sample_length, rng)
        buckets = buckets + [name] * sample_length
    data[metric_name] = metric
    data[bucket_column] = buckets
    raw_data = pd.DataFrame.from_dict(data)
    return raw_data


def create_continuous_data(means, stds, bucket_names, metric_name, log=True,
                           bucket_column='bucket', sample_length=1000, seed=0):
    """Create a dataframe to simulate conversion data.

    means, stds, and bucket_names should be of length = number of variants.

    Arguments:
        means {list} -- list of the means of the variants
        stds {list} -- list of the standard deviations of the variants
        bucket_names {list} -- list strings of the bucket names of the
            variants
        metric_name {str} -- string for the name of the continuous metric
        log {bool} -- should this be lognormal data {default: {True}}
        bucket_column {str} -- string for the name of the column for the
            buckets
        sample_length {int} number of obersvations to be in each sample
            data (default: {1000})
        seed {int} -- seed for the data. One generator is shared by all the
            buckets (default: {0})

    Returns:
        raw_data {dataframe} -- dataframe
    """
    rng = np.random.default_rng(seed)
    data = {}
    metric = []
    buckets = []
    for mean, std, name in zip(means, stds, bucket_names):
        metric = metric + create_continuous_sample(mean,
                                                   std,
                                                   log,
                                                   sample_length,
                                                   rng)
        buckets = buckets + [name] * sample_length
    data[metric_name] = metric
    data[bucket_column] = buckets
    raw_data = pd.DataFrame.from_dict(data)
    return raw_data


def create_poisson_data(means, bucket_names, metric_name,
                        bucket_column='bucket', sample_length=1000, seed=0):
    """Create a dataframe to simulate poisson/count data.

    means and bucket_names should be of length = number of variants.

    Arguments:
        means {list} -- list of the means of the variants
        bucket_names {list} -- list strings of the bucket names of the
            variants
        metric_name {str} -- string for the name of the poisson metric
        bucket_column {str} -- string for the name of the column for the
            buckets
        sample_length {int} number of obersvations to be in each sample
            data (default: {1000})
        seed {int} -- seed for the data. One generator is shared by all the
            buckets (default: {0})

    Returns:
        raw_data {dataframe} -- dataframe
    """
    rng = np.random.default_rng(seed)
    data = {}
    metric = []
    buckets = []
    for mean, name in zip(means, bucket_names):
        metric = metric + create_poisson_sample(mean, sample_length, rng)
        buckets = buckets + [name] * sample_length
    data[metric_name] = metric
    data[bucket_column] = buckets
    raw_data = pd.DataFrame.from_dict(data)
    return raw_data
