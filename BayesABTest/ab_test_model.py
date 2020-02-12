import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from itertools import combinations
from ._ab_test_model_plotting import _ab_test_plotting
from ._ab_test_model_distributions import _ab_test_distributions
from ._ab_test_model_loss_func import _ab_test_loss_functions

# empirical bayes method https://en.wikipedia.org/wiki/Empirical_Bayes_method
# https://juanitorduz.github.io/intro_pymc3/
# https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.Gamma
# https://docs.pymc.io/api/distributions/discrete.html#pymc3.distributions.discrete.Poisson

class ab_test_model(_ab_test_plotting, _ab_test_distributions, _ab_test_loss_functions):
  """
  DESCRIPTION:
   This class fits an ab_test_model object to your data, and will output the results of the test.
   This is a easy to use wrapper around Pymc3, with built in plotting and reporting.

  METHODS:
    fit()
      requires-- class was instantiated with valid inputs
      modifies-- ecdf, control_sample, variant_sample, lift
      effects-- creates and runs the monte carlo simulation, sets member variables to reflect model outputs

    plot()
      requires-- fit() has been run
      modifies-- none
      effects-- creates a 3 chart report of the AB test and opens it in a python viewer

  INSTANTIATION VARIABLES:
     raw_data -- a pandas dataframe with (at least) two columns, one for the bucket and one for the response variable
     metric -- column name in raw_data for the response variable
     bucket_col_name -- column name in raw_data for the bucket (defaults to bucket)
     control_bucket_name -- value in bucket_col_name for the control bucket (defaults to off)
     variant_bucket_name -- value in bucket_col_name for the variant bucket (defaults to on)
     samples -- number of samples to run the monte carlo simulation, must be 1,000 and 50,000 (defaults to 10,000)
     prior_func -- the type of distribution to use for the prior. options include:
       beta -- use for conversion rates. bounded on the interval [0,1]
       log--normal -- use for continuous, greater than zero response variables (0, +inf) (ex: premium, minutes spent on scene, etc.)
       normal -- use for continuous response variables that on the interval (-inf, +inf)
       poisson -- use for discrete, greater than zero response variables (ex: arrivals per day, events per account, etc.)
     prior_info -- the prior belief to about the response varable. options include:
       uninformed -- no prior belief about the response, all outcomes are equaly likely
       informed -- uses the control as the prior belief, will add support to scale down that belief to a weaker belief
     confidence_level -- value for the confidence interval on the CDF chart (defaults to 0.05)
     compare_variants -- boolean for comparing the variants to each other. Control to each variant is always done (unless there are too many variants to plot). If there are few enough variants, the comparisons for variants will be plotted. (defaults to False)
     debug -- boolean to print out extra output for debugging purposes (defaults to False)
     prior_scale_factor -- factor to scale an informed prior by (see empirical [empirical bayes](https://en.wikipedia.org/wiki/Empirical_Bayes_method)) (defaults to 4)

     RUNTIME VARAIBLES:
      ecdf -- a dictionary with keys x and y, and the value is a list containing the x and y coordinates for the empirical CDF (only meaningful after fit() is run)
      control_sample -- a list containing the sampled values from the control posterior distribution (only meaningful after fit() is run)
      variant_sample -- a list containing the sampled values from the variant posterior distribution (only meaningful after fit() is run)
      lift -- a list containing the lift of the variant over the control for each sampled value (only meaningful after fit() is run)
     """
  def __init__(self, raw_data, metric, samples=10000, prior_func='beta',
               prior_info='informed', control_bucket_name='off',
               variant_bucket_names=['on'], bucket_col_name='bucket',
               confidence_level=.05, compare_variants=False, debug=False,
               prior_scale_factor=4):
    """
    Initialize an instance of BayesABTest with input variables.
    See class doc string for variable explanation.
    """
    # run input checking
    self._error_check_inputs(raw_data, metric, samples, prior_func,
                              prior_info, control_bucket_name,
                              variant_bucket_names, bucket_col_name,
                              confidence_level, compare_variants,
                              debug, prior_scale_factor)

    # PUBLIC VARIABLES
    self.debug = debug
    self.prior_info = prior_info
    self.prior_func = prior_func
    self.raw_data = raw_data
    self.metric = metric
    self.samples = samples
    self.compare_variants = compare_variants
    if samples * (1 + len(variant_bucket_names)) > 50000:
      print('WARNING: This is a large amount of sampling. Run time may be long.')
      print('Running', samples, 'on', 1 + len(variant_bucket_names),
            'PDFs, for a total of', samples * (1 + len(variant_bucket_names)),
            'samples')
    self.control_bucket_name = control_bucket_name
    if not isinstance(variant_bucket_names, list):
      self.variant_bucket_names = list(variant_bucket_names)
    else: self.variant_bucket_names = variant_bucket_names
    self.bucket_col_name = bucket_col_name
    self.confidence_level = confidence_level
    self.ecdf = {}
    self.prior_scale_factor = prior_scale_factor
    self.control_sample = []
    self.variant_samples = []
    self.lift = []

    self._trace = None
    self._buckets = self.raw_data[self.bucket_col_name].unique()

  def _error_check_inputs(self, raw_data, metric, samples, prior_func,
                           prior_info, control_bucket_name,
                           variant_bucket_names, bucket_col_name,
                           confidence_level, compare_variants,
                           debug, prior_scale_factor):
    """Check the input variables for errors. This class only supports
    the following inputs.
    """
    SUPPORTED_PRIOR_FUNC = ['beta', 'log-normal', 'normal', 'poisson'] # to do: naming is wrong cuz some are priors some are posts
    SUPPORTED_PRIOR_INFO = ['informed', 'uninformed']

    if raw_data.empty: raise Exception('Input dataframe must contain data')
    if prior_func == 'beta' and ([np.any(x) for x in raw_data[metric].unique() if x not in list([0,1])] or [False])[0]:
      raise Exception('Observed data for beta prior should be binary')
    if metric not in raw_data.columns:
      raise Exception('Input dataframe must contain column:', metric)
    if samples < 1000 or samples > 50000:
      raise Exception('Number of samples must be in the interval [1000,50000]')
    if prior_func not in SUPPORTED_PRIOR_FUNC:
      raise Exception('Prior must be' +
                      'in [{}]'.format(', '.join(SUPPORTED_PRIOR_FUNC)))
    if prior_info not in SUPPORTED_PRIOR_INFO:
      raise Exception('Prior must be in' +
                      '[{}]'.format(', '.join(SUPPORTED_PRIOR_INFO)))
    if control_bucket_name not in raw_data[bucket_col_name].unique():
      raise Exception('Input dataframe', bucket_col_name,
                      'column must contain values with:', control_bucket_name,
                      'as the control bucket name')
    if compare_variants and len(variant_bucket_names) <  2:
      raise Exception('If compare_variants is True, there must be at least 2 variants')
    if len(variant_bucket_names) > 10:
      raise Exception('Greater than 10 variants is not currently supported')
    for name in variant_bucket_names:
      if name not in raw_data[bucket_col_name].unique():
        raise Exception('Input dataframe', bucket_col_name,
                        'column must contain values with:', name,
                        'as a variant bucket name')
    for name in raw_data[bucket_col_name].unique():
      if name not in variant_bucket_names and name != control_bucket_name:
        raise Exception('Input dataframe contains value ', name, 'in',
                        bucket_col_name,
                        'that is not a variant name in variant_bucket_names')
    if bucket_col_name not in raw_data.columns:
      raise Exception('Input dataframe must contain column:', bucket_col_name)
    if confidence_level <= 0 or confidence_level >= 1:
      raise Exception('Confidence level must be in the interval (0,1)')
    if debug not in [True, False]:
      raise Exception('debug must be either True or False')
    if compare_variants not in [True, False]:
      raise Exception('compare_variants must be either True or False')


  def fit(self):
    """Set up the pymc3 model with the prior parameters."""
    model = pm.Model()
    with model:
      self._set_distributions()

    if self.debug:
      print('Running', self.samples, 'MCMC simulations')
    with model:
      self._trace = pm.sample(self.samples)
      self._set_samples()

    if self.debug:
      summary = pm.summary(self._trace)
      print('Posterior sampling distribution summary statistics' +
            '\n {}'.format(summary[['mean', 'sd']].round(4)))

    self._calc_lift()
    self._calc_ecdf()


  def plot(self, lift_plot_flag=True):
    """Utilize the private graphing functions to
    create a report of the test in one image.

    Params:
      lift_plot_flag -- boolean for plotting lift PDF and CDF (defaults to True)
    """
    if self.control_sample == []: raise Exception('fit() must be' +
                                                  'run before plot')

    if lift_plot_flag and len(self.variant_bucket_names) > 3 and self.compare_variants:
      lift_plot_flag = False
      print('WARNING: lift_plot_flag reset to False due to the large number of combinations of variant comparisons.')
      print('Re-run with less variants selected for comparison, or turn off compare_variants')
    if len(self.variant_bucket_names) > 5:
      lift_plot_flag = False
      print('WARNING: lift PDF and CDF graphing is turned off for more than 5 varaints. Only the posterior PDFs will be plotted.')
      print('Re-run with less variants selected for lift comparisons, or run with debug=True to see relative lifts in command line.')

    fig = plt.figure(figsize=[12.8, 9.6])
    fig.suptitle('Bayesian AB Test Report for {}'.format(self.metric),
                                                         fontsize=16,
                                                         fontweight='bold')
    fig.subplots_adjust(hspace=1)

    # PDFS ONLY
    if not lift_plot_flag:
      self._plot_posteriors()
      plt.show()

    # ONE VARIANT
    elif len(self.variant_bucket_names) == 1:
      plt.subplot(2,2,1)
      lift = self.lift[0]
      variant_name = self.variant_bucket_names[0]
      self._plot_posteriors()
      plt.subplot(2,2,2)
      self._plot_positive_lift(lift)
      plt.subplot(2,1,2)
      self._plot_ecdf(variant_name)
      plt.show()

    # MULTIPLE VARIANTS
    else:
      nrows = len(self.variant_bucket_names) + 1
      if self.compare_variants:
        comparisons = list(range(0,len(self.variant_bucket_names)))
        combs = list(combinations(comparisons, 2))
        nrows += len(combs)
      plt.subplot(nrows,1,1)
      self._plot_posteriors()
      loc = 3 # set initial figure location
      for num in range(0, len(self.variant_bucket_names)):
        plt.subplot(nrows,2,loc)
        self._plot_positive_lift(self.lift[num], sample1=-1, sample2=num)
        plt.subplot(nrows,2,loc+1)
        self._plot_ecdf(self.variant_bucket_names[num])
        loc += 2
      if self.compare_variants:
        for num in range(0, len(combs)):
          plt.subplot(nrows,2,loc)
          self._plot_positive_lift(self.lift[num+len(self.variant_bucket_names)], sample1=combs[num][1],
                                    sample2=str(combs[num][0]))
          plt.subplot(nrows,2,loc+1)
          name = 'bucket_comparison' + \
                  str(combs[num][1]) + '_' + str(combs[num][0])
          self._plot_ecdf(name)
          loc += 2
      plt.show()
