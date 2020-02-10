import pandas as pd
import numpy as np

## GENERATE DATA HELPER FUNCTIONS
# This file creates random data for use in testing
def create_conversion_sample(true_rate, length=1000):
  """Create a conversion binary list of n=length samples with true_rate."""
  sample = list(np.random.rand(length))
  for index, prob in enumerate(sample):
    if prob < true_rate: sample[index] = 1
    else: sample[index] = 0
  return sample

def create_continuous_sample(mean, sigma, log=True, length=1000):
  """Create a random (log) normal list of n=length samples with mean and sigma.
  Input values should not be logged."""
  if log:
    mean = np.log(mean)
    sigma = np.log(sigma)
    sample = list(np.random.lognormal(mean=mean, sigma=sigma, size=length))
  else:
    sample = list(np.random.normal(loc=mean, scale=sigma, size=length))
  return sample

def create_conversion_data(true_rates, bucket_names, metric_name,
                           bucket_column='bucket', sample_length=1000):
  """Create a random dataframe to simulate conversion data.
  true_rates and bucket_names should be of length = number of variants.
  returns a dataframe."""
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
  """Create a random dataframe to simulate continuous data.
  means, stds, and bucket_names should be of length = number of variants.
  returns a dataframe."""
  data = {}
  metric = []
  buckets = []
  for mean, std, name in zip(means, stds, bucket_names):
    metric = metric + create_continuous_sample(mean, std, log, sample_length)
    buckets = buckets + [name] * sample_length
  data[metric_name] = metric
  data[bucket_column] = buckets
  raw_data = pd.DataFrame.from_dict(data)
  return raw_data

def create_poisson_sample(mean, length=1000):
  """Create a conversion binary list of n=length samples with true_rate."""
  sample = list(np.random.poisson(lam=mean, size=length))
  return sample

def create_poisson_data(means, bucket_names, metric_name,
                           bucket_column='bucket', sample_length=1000):
  """Create a random dataframe to simulate poisson data.
  means, and bucket_names should be of length = number of variants.
  returns a dataframe."""
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