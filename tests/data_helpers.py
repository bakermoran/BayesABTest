"""Functions to help generate fake test data."""

import pandas as pd
import numpy as np


def create_conversion_sample(true_rate, length=1000):
    """Create a conversion binary list of n=length samples with true_rate.

    Arguments:
        true_rate {float} - true conversion rate of the sample. on the
            interval [0,1].
        length {int} -- number of obersvations to be in the sample data
            (default: {1000})

    Returns:
        sample {list} -- list of conversion samples
    """
    sample = list(np.random.rand(length))
    for index, prob in enumerate(sample):
        if prob < true_rate:
            sample[index] = 1
        else:
            sample[index] = 0
    return sample


def create_continuous_sample(mean, sigma, log=True, length=1000):
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
        sample = list(np.random.lognormal(mean=mean, sigma=sigma, size=length))
    else:
        sample = list(np.random.normal(loc=mean, scale=sigma, size=length))
    return sample


def create_poisson_sample(mean, length=1000):
    """Create a random poisson list of n=length samples with mean.

    Arguments:
        Input values should not be logged.
        mean {float} -- mean value of the poisson distribution
        length {int} -- number of obersvations to be in the sample
            data (default: {1000})

    Returns:
        sample {list} -- list of poisson samples
    """
    sample = list(np.random.poisson(lam=mean, size=length))
    return sample


def create_conversion_data(true_rates, bucket_names, metric_name,
                           bucket_column='bucket', sample_length=1000):
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

    Returns:
        raw_data {dataframe} -- dataframe
    """
    data = {}
    metric = []
    buckets = []
    for rate, name in zip(true_rates, bucket_names):
        metric = metric + create_conversion_sample(rate, sample_length)
        buckets = buckets + [name] * sample_length
    data[metric_name] = metric
    data[bucket_column] = buckets
    raw_data = pd.DataFrame.from_dict(data)
    return raw_data


def create_continuous_data(means, stds, bucket_names, metric_name, log=True,
                           bucket_column='bucket', sample_length=1000):
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

    Returns:
        raw_data {dataframe} -- dataframe
    """
    data = {}
    metric = []
    buckets = []
    for mean, std, name in zip(means, stds, bucket_names):
        metric = metric + create_continuous_sample(mean,
                                                   std,
                                                   log,
                                                   sample_length)
        buckets = buckets + [name] * sample_length
    data[metric_name] = metric
    data[bucket_column] = buckets
    raw_data = pd.DataFrame.from_dict(data)
    return raw_data


def create_poisson_data(means, bucket_names, metric_name,
                        bucket_column='bucket', sample_length=1000):
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

    Returns:
        raw_data {dataframe} -- dataframe
    """
    data = {}
    metric = []
    buckets = []
    for mean, name in zip(means, bucket_names):
        metric = metric + create_poisson_sample(mean, sample_length)
        buckets = buckets + [name] * sample_length
    data[metric_name] = metric
    data[bucket_column] = buckets
    raw_data = pd.DataFrame.from_dict(data)
    return raw_data
