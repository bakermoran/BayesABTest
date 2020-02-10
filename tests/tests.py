import pandas as pd
from BayesABTest import BayesABTest as ab
import data_helpers as dh

## TEST FUNCTIONS
# this file runs some sample tests to check if output is working
def one_variant_conversion():
  """Create data and run a one variant report for a conversion metric."""
  raw_data_auto_bind = dh.create_conversion_data([.27,.3],['off','on'],
                                              metric_name='bind')
  auto_bind = ab.BayesABTest(raw_data_auto_bind, metric='bind',
                          prior_info='uninformed', prior_func='beta', debug=True,
                          samples=1000)
  auto_bind.create_model()
  auto_bind.create_report()
  auto_bind.loss_function()


def one_variant_continuous():
  """Create data and run a one variant report for a continuous metric."""
  # test between off and on buckets (small difference)
  rawdata_small = dh.create_continuous_data([600,610],[1.5,1.5],['off','on'],
                                         metric_name='total_premium')
  premium_small = ab.BayesABTest(rawdata_small, metric='total_premium',
                  prior_info='informed', prior_func='log-normal', debug=True,
                  samples=1000)
  premium_small.create_model()
  premium_small.create_report()

 # test between off and on buckets (big difference)
  rawdata_big = dh.create_continuous_data([600,700],[1.5,1.5],['off','on'],
                                       metric_name='total_premium')
  premium_big = ab.BayesABTest(rawdata_big, metric='total_premium',
                  prior_info='informed', prior_func='log-normal', debug=True,
                  samples=1000)
  premium_big.create_model()
  premium_big.create_report()

def one_variant_continuous_small_mean():
  """Create data and run a one variant report for a continuous metric
  with a mean between 0 and 1."""
  rawdata_small = dh.create_continuous_data([.65,.66],[1.5,1.5],['off','on'],
                                         metric_name='loss_ratio')
  premium_small = ab.BayesABTest(rawdata_small, metric='loss_ratio',
                  prior_info='informed', prior_func='log-normal', debug=True,
                  samples=1000)
  premium_small.create_model()
  premium_small.create_report()

def two_variants_conversion():
  """Create data and run a two variant report for a conversion metric."""
  raw_data_2vars = dh.create_conversion_data([.22,.23,.235],
                                          ['control','variant_1','variant_2'],
                                          metric_name='bind')
  auto_bind = ab.BayesABTest(raw_data_2vars, metric='bind',
                            prior_info='uninformed', prior_func='beta',
                            debug=True, control_bucket_name='control',
                            variant_bucket_names=['variant_1','variant_2'],
                            compare_variants=True, lift_plot_flag=True,
                            samples=1000)
  auto_bind.create_model()
  auto_bind.create_report()

def conversion_negative_variants():
  """Create data and run a two variant report for a conversion metric
  where the variants are worse than the control."""
  raw_data_2vars = dh.create_conversion_data([.25,.23,.235],
                                          ['control','variant_1','variant_2'],
                                          metric_name='bind')
  auto_bind = ab.BayesABTest(raw_data_2vars, metric='bind',
                            prior_info='uninformed', prior_func='beta',
                            debug=True, control_bucket_name='control',
                            variant_bucket_names=['variant_1','variant_2'],
                            compare_variants=True, lift_plot_flag=True,
                            samples=1000)
  auto_bind.create_model()
  auto_bind.create_report()

def two_variants_continuous():
  """Create data and run a two variant report for a continuous metric."""
  rawdata = dh.create_continuous_data([600,610,615],[1.5,1.5,1.5],
                                   ['control','variant_1','variant_2'],
                                   metric_name='total_premium')
  premium = ab.BayesABTest(rawdata, metric='total_premium',
                            prior_info='informed', prior_func='log-normal',
                            debug=True, control_bucket_name='control',
                            variant_bucket_names=['variant_1','variant_2'],
                            compare_variants=True,
                            samples=1000)
  premium.create_model()
  premium.create_report()

def three_variants_continuous():
  """Create data and run a three variant report for a continuous metric."""
  rawdata = dh.create_continuous_data([600,610,615,620],[1.5,1.5,1.5,1.5],
                                   ['control','variant_1','variant_2','variant_3'],
                                   metric_name='total_premium')
  premium = ab.BayesABTest(rawdata, metric='total_premium',
                            prior_info='informed', prior_func='log-normal',
                            debug=True, control_bucket_name='control',
                            variant_bucket_names=['variant_1','variant_2','variant_3'],
                            compare_variants=False,
                            samples=1000)
  premium.create_model()
  premium.create_report()

def four_variants_continuous():
  """Create data and run a four variant report for a continuous metric."""
  rawdata = dh.create_continuous_data([600,610,615,620,625],[1.5,1.5,1.5,1.5,1.5],
                                   ['control','variant_1','variant_2','variant_3','variant_4'],
                                   metric_name='total_premium')
  premium = ab.BayesABTest(rawdata, metric='total_premium',
                            prior_info='informed', prior_func='log-normal',
                            debug=True, control_bucket_name='control',
                            variant_bucket_names=['variant_1','variant_2','variant_3','variant_4'],
                            compare_variants=False, lift_plot_flag=True,
                            samples=1000)
  premium.create_model()
  premium.create_report()

def two_variants_continuous_normal():
  """Create data and run a four variant report for a normal
  continuous metric."""
  rawdata = dh.create_continuous_data([600,601,602],[30,30,30],
                                   ['control','variant_1','variant_2'],
                                   metric_name='total_premium', log=False)
  premium = ab.BayesABTest(rawdata, metric='total_premium',
                            prior_info='informed', prior_func='normal',
                            debug=True, control_bucket_name='control',
                            variant_bucket_names=['variant_1','variant_2'],
                            compare_variants=True, lift_plot_flag=True,
                            samples=1000)
  premium.create_model()
  premium.create_report()

def two_variants_poisson():
  """Create data and run a four variant report for a normal
  poisson metric."""
  rawdata = dh.create_poisson_data([15,17,20],
                                   ['control','variant_1','variant_2'],
                                   metric_name='vists', sample_length=3000)

  vists = ab.BayesABTest(rawdata, metric='vists',
                            prior_info='informed', prior_func='poisson',
                            debug=True, control_bucket_name='control',
                            variant_bucket_names=['variant_1','variant_2'],
                            compare_variants=True, lift_plot_flag=True,
                            samples=3000)
  vists.create_model()
  vists.create_report()

def test_controller():
  # one_variant_conversion()
  # one_variant_continuous()
  # one_variant_continuous_small_mean()
  # two_variants_conversion()
  # conversion_negative_variants()
  # two_variants_conversion()
  # two_variants_continuous()
  # two_variants_continuous_normal()
  # three_variants_continuous()
  # four_variants_continuous()
  two_variants_poisson()

if __name__ == "__main__":
  test_controller()
