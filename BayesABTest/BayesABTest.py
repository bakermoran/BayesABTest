import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special as sc
from scipy import stats
from itertools import combinations
#pylint: disable=no-member

# to do: log normal and beta 'get prior' and 'get posteriors' should have the
  # same interface, but they dont
  # they're private right now so it doesnt matter too much for the time being
# to do: get log normal uninformed prior right
# to do: on the cdf plot, dont despine the left (when i try this it
  # applies to all of them)
# to do: allow a user to specify the prior
# to do: separate posterior generation from the inputs. make it just a function
  # that takes data
# to do: get everything 79 chars or less on one line
# to do: add unit tests
# to do: add ability to not sample, but run with seeded random numbers
  # (scipy or numpy)
# to do: break package code in to separate files
# to do: allow someone to input a lot of data, but select out two
  # variants to plot
    # ex: data contains var1, var2, var3, but want to only test var1 and var3
    # for some reason, this doesnt work:
      # self.raw_data =
        # raw_data.loc[raw_data[bucket_col_name].isin(variant_bucket_names)]
  # or else do this for now: make sure number of buckets in dataframe match
    # number of buckets in variant_bucket_names
# to do: debug should actually output diagnostic info of the model
# to do: add public interface to the single graphing functions, with args for variant names
# to do: lift_plot_flag and compare_variants should be an input to the create report function

# empirical bayes method https://en.wikipedia.org/wiki/Empirical_Bayes_method
# https://juanitorduz.github.io/intro_pymc3/
# https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.Gamma
# https://docs.pymc.io/api/distributions/discrete.html#pymc3.distributions.discrete.Poisson

class BayesABTest:
  def __init__(self, raw_data, metric, samples=10000, prior_func='beta',
               prior_info='informed', control_bucket_name='off',
               variant_bucket_names=['on'], bucket_col_name='bucket',
               confidence_level=.05, compare_variants=False, debug=False,
               lift_plot_flag=True, prior_scale_factor=4):
    """Initialize an instance of BayesABTest with input variables."""
    # run input checking
    self.__error_check_inputs(raw_data, metric, samples, prior_func,
                              prior_info, control_bucket_name,
                              variant_bucket_names, bucket_col_name,
                              confidence_level, compare_variants,
                              debug, lift_plot_flag, prior_scale_factor)

    # PUBLIC VARIABLES
    self.debug = debug
    if lift_plot_flag and len(variant_bucket_names) > 3 and compare_variants:
      self.lift_plot_flag = False
      print('WARNING: lift_plot_flag reset to False due to the large number of combinations of variant comparisons.')
      print('Re-run with less variants selected for comparison, or turn off compare_variants')
    self.compare_variants = compare_variants
    if len(variant_bucket_names) > 5:
      self.lift_plot_flag = False
      print('WARNING: lift PDF and CDF graphing is turned off for more than 5 varaints. Only the posterior PDFs will be plotted.')
      print('Re-run with less variants selected for lift comparisons, or run with debug=True to see relative lifts in command line.')
    else: self.lift_plot_flag = lift_plot_flag
    self.prior_info = prior_info
    self.prior_func = prior_func
    self.raw_data = raw_data
    self.metric = metric
    self.samples = samples
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

    # PRIVATE VARIABLES
    self.__trace = None
    self.__buckets = self.raw_data[self.bucket_col_name].unique()

  # PRIVATE METHODS
  def __error_check_inputs(self, raw_data, metric, samples, prior_func,
                           prior_info, control_bucket_name,
                           variant_bucket_names, bucket_col_name,
                           confidence_level, compare_variants,
                           debug, lift_plot_flag, prior_scale_factor):
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
    if debug not in [True, False] or \
       compare_variants not in [True, False] or \
       lift_plot_flag not in [True, False]:
      raise Exception('debug must be either True or False')


  # LOSS FUNCTIONS
  def __h(self, a, b, c, d):
    sum = 0
    for j in range(0,c):
      w = sc.betaln(a+j, b+d)
      x = np.log(d + j)
      y = sc.betaln(1 + j, d)
      z = sc.betaln(a,b)
      sum += np.exp(w-x-y-z)
    print(sum)
    return 1 - sum


  def __loss_beta(self, a, b, c, d, loss_type):
    if loss_type == 'absolute':
      x1 = sc.betaln(a+1,b)
      y1 = np.log(self.__h(a+1,b,c,d))
      z1 = sc.betaln(a,b)

      x2 = sc.betaln(c+1,d)
      y2 = np.log(self.__h(a,b,c+1,d))
      z2 = sc.betaln(c,d)

      return np.exp(x1 + y1 - z1) - np.exp(x2 + y2 - z2)


    elif loss_type == 'percent':
      prob_1 = self.__h(a,b,c,d)

      x = sc.betaln(a - 1, b)
      y = sc.betaln(a, b)
      z = sc.betaln(c + 1, d)
      w = sc.betaln(c, d)
      prob_2 = np.log(self.__h(a - 1, b, c + 1, d))

      return prob_1 - np.exp(x - y + z - w + prob_2)


  def __loss_gamma(self, a, b, c, d):
    x1 = sc.gammaln(a+1)
    y1 = np.log(self.__h(a+1,b,c,d))
    z1 = np.log(b)
    w1 = sc.gammaln(a)

    x2 = sc.gammaln(c+1)
    y2 = np.log(self.__h(a,b,c+1,d))
    z2 = np.log(d)
    w2 = sc.gammaln(c)

    return np.exp(x1 + y1 - z1 - w1) - np.exp(x2 + y2 - z2 - w2)


  def __loss_normal(self, sample1, sample2):
    diffs = []
    for x, y in zip(sample1, sample2):
      diffs.append(max(x-y), 0)
    return np.mean(diffs)


  def __stop_test(self, risk_A, risk_B, epsilon):
    if risk_A <= epsilon or risk_B <= epsilon:
      if risk_A <= epsilon: return True, self.control_bucket_name
      else: return True, self.variant_bucket_names[0]
    else: return False, None


  # GRAPHING MODULE
  def __format_axis_as_percent(self, locs, labels):
      labels = []
      for i in range(len(locs)):
        labels.append('{:.0%}'.format(locs[i]))
      return labels

  def __stringify_variants(self):
    strings = [self.control_bucket_name] + self.variant_bucket_names
    last_word = []
    if len(strings) == 2:
      title = strings[0] + ' and ' + strings[1]
    else:
      last_word = strings[-1]
      strings[-1] = 'and '
      title = ', '.join(strings)
      title += str(last_word)
    return title

  def __format_title(self, string):
    length = len(string)
    if length < 40: return string
    newline = False
    for index, char in enumerate(string):
      if index < 40: continue
      if index % 40 == 0 and not newline:
        newline = True
        continue
      if newline and char == ' ':
        string = list(string)
        string[index] = '\n'
        string = ''.join(string)
        newline = False
    return string


  def __plot_posteriors(self):
    """Plot kernel density estimations of the posterior samples."""
    colors = ['blue', 'red', 'green', 'yellow', 'purple', 'cyan']
    colors += colors
    sns.kdeplot(self.control_sample, shade=True, color=colors[0])
    for i in range(0, len(self.variant_bucket_names)):
      sns.kdeplot(self.variant_samples[i], shade=True, color=colors[i+1])
    plt.legend(labels=self.__buckets, loc='upper right')
    if self.prior_func == 'beta': plt.xlabel('Conversion Rate')
    elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
      plt.xlabel(self.metric)
    sns.despine(left=True)
    plt.yticks([],[])
    title = 'Distributions for {0} for {1}'.format(self.__stringify_variants(),
                                                   self.metric)
    title = self.__format_title(title)
    plt.title(title)
    if self.prior_func == 'beta':
      locs, labels = plt.xticks()
      labels = self.__format_axis_as_percent(locs, labels)
      plt.xticks(locs, labels=labels)


  def __plot_positive_lift(self, lift, sample1=None, sample2=None):
    """Plot the lift as a kernel density estimation.
    sample1 and sample2 are for the row title. They are not used for
    one variant, and for multiple variants, sample 1 is the denominator,
    sample2 is the numerator. -1 for sample1 means it is the control"""
    ax = sns.kdeplot(lift,shade=True)
    line = ax.get_lines()[-1]
    x, y = line.get_data()
    ylim = max(y)*1.1
    mask = x > 0
    x, y = x[mask], y[mask]
    ax.fill_between(x, y1=y, alpha=0.5, facecolor='red')
    if len(self.variant_bucket_names) > 1:
      sample1 = int(sample1)
      sample2 = int(sample2)
      if sample1 == -1:
        denom_name = self.control_bucket_name
      else: denom_name = self.variant_bucket_names[sample1]
      num_name = self.variant_bucket_names[sample2]
      title = num_name + ' vs ' + denom_name
      ax.set_ylabel(title, rotation=0, fontstyle='italic')
    plt.vlines(0.0,ymin=0,ymax=ylim,linestyle='dotted')
    plt.xlabel('Lift')
    percent_positive_lift = sum(i > 0 for i in lift) / len(lift)
    title = '{0} had {1:.2%} probability of positive lift'.format(self.metric,
                                                         percent_positive_lift)
    title = self.__format_title(title)
    plt.title(title)
    sns.despine(left=True)
    plt.yticks([],[])
    locs, labels = plt.xticks()
    labels = self.__format_axis_as_percent(locs, labels)
    plt.xticks(locs, labels=labels)


  def __plot_ecdf(self, variant_name):
    """Plot the empirical cumulative distribution function."""
    x = self.ecdf[variant_name]['x']
    y = self.ecdf[variant_name]['y']

    lower_bound = x[y.index(min(y, key=lambda x:abs(x-self.confidence_level)))]
    median = x[y.index(min(y, key=lambda x:abs(x-0.5)))]
    upper_bound = x[y.index(min(y,
                            key=lambda x:abs(x-(1-self.confidence_level))))]

    sns.lineplot(x=self.ecdf[variant_name]['x'], y=self.ecdf[variant_name]['y'])
    ci = 1 - self.confidence_level
    title = 'Median Lift was {0:.2%}, with a {1:.0%} CI of [{2:.2%}, {3:.2%}]'.format(median,
                                                                                      ci,
                                                                                      lower_bound,
                                                                                      upper_bound)
    title = self.__format_title(title)
    plt.title(title)
    plt.xlabel('Lift')
    plt.ylabel('Cumulative Probability')
    plt.vlines(lower_bound,ymin=0,ymax=1,linestyle='dotted')
    plt.vlines(median,ymin=0,ymax=1,linestyle='dotted')
    plt.vlines(upper_bound,ymin=0,ymax=1,linestyle='dotted')
    sns.despine(left=True)
    locs, labels = plt.xticks()
    labels = self.__format_axis_as_percent(locs, labels)
    plt.xticks(locs, labels=labels)


  def __calc_ecdf(self):
    """Calculate the empirical cumulative distribution function
    and set member var.
    """
    if self.compare_variants:
      comparisons = list(range(0,len(self.variant_bucket_names)))
      combs = list(combinations(comparisons, 2))
    num_variants = len(self.variant_bucket_names)
    for index, lift in enumerate(self.lift):
      raw_data = np.array(lift)
      cdfx = np.sort(np.unique(self.lift))
      x_values = np.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))
      size_data = raw_data.size
      y_values = []
      for i in x_values:
        temp = raw_data[raw_data <= i]
        value = temp.size / size_data
        y_values.append(value)
      temp = {}
      temp['x'] = x_values
      temp['y'] = y_values
      if index < num_variants:
        self.ecdf[self.variant_bucket_names[index]] = temp
      elif self.compare_variants and index >= num_variants:
        name = 'bucket_comparison' + str(combs[index-num_variants][1]) + \
                '_' + str(combs[index-num_variants][0])
        self.ecdf[name] = temp


  def __calc_lift(self):
    """Calculate the lift of the variant over the control
    of the posterior samples.
    """
    for index, sample in enumerate(self.variant_samples):
      self.lift.append(sample / self.control_sample - 1)
      if self.debug:
        percent_positive_lift = sum(i > 0 for i in self.lift[index]) / len(self.lift[index])
        print('percent positive lift for {0} over {1} = {2:.2%}'.format(self.variant_bucket_names[index],
                                                                        self.control_bucket_name,
                                                                        percent_positive_lift))

    if self.compare_variants:
      comparisons = list(range(0,len(self.variant_bucket_names)))
      combs = combinations(comparisons, 2)
      for combination in combs:
        self.lift.append(np.array(self.variant_samples[combination[1]]) / np.array(self.variant_samples[combination[0]]) - 1)
        if self.debug:
          percent_positive_lift = sum(i > 0 for i in self.lift[-1]) / len(self.lift[-1])
          print('percent positive lift for {0} over {1} = {2:.2%}'.format(self.variant_bucket_names[combination[1]],
                                                                          self.variant_bucket_names[combination[0]],
                                                                          percent_positive_lift))


  # DISTRIBUTIONS MODULE
  def __get_beta_priors(self, raw_data_agg):
    """Return prior distribution of a beta distribution,
    either informed or uninformed depending on input.
    """
    num_vars = 1 + len(self.variant_bucket_names)
    if self.prior_info == 'informed':
      prior_converts = raw_data_agg.loc[self.control_bucket_name][self.metric]
      prior_non_converts = raw_data_agg.loc[self.control_bucket_name] \
                                           ['bucketed'] - prior_converts
      prior_distribution = pm.Beta(self.metric,
                                   alpha=prior_converts/self.prior_scale_factor,
                                   beta=prior_non_converts/self.prior_scale_factor,
                                   shape=num_vars)
    elif self.prior_info == 'uninformed':
      prior_distribution = pm.Beta(self.metric, alpha=1, beta=1, shape=num_vars)
    return prior_distribution

  def __get_poisson_priors(self, raw_data_agg):
    """Return prior distribution of a gamma distribution,
    either informed or uninformed depending on input.
    """
    num_vars = 1 + len(self.variant_bucket_names)
    if self.prior_info == 'informed':
      mu = np.mean(raw_data_agg[self.metric].values)
      sigma2 = np.mean(raw_data_agg['variance'].values)
      sigma2 *= self.prior_scale_factor # weaken the prior

      alpha = mu**2/sigma2
      beta = mu/sigma2

      prior_distribution = pm.Gamma(self.metric,
                                    alpha=alpha,
                                    beta=beta,
                                    shape=num_vars)
    elif self.prior_info == 'uninformed':
      raise Exception('poisson uninformed prior not yet implemented')
    return prior_distribution


  def __get_normal_priors(self, control_data):
    """Return prior distribution parameters of a (log)normal
    distribution either informed or uninformed depending on input.
    """
    if self.prior_info == 'informed':
      if self.prior_func == 'log-normal':
        prior_mean = np.mean(np.log(control_data))
        prior_var = np.var(np.log(control_data))
        prior_var *= self.prior_scale_factor
      elif self.prior_func == 'normal':
        prior_mean = np.mean(control_data)
        prior_var = np.var(control_data)
        prior_var *= self.prior_scale_factor
    elif self.prior_info == 'uninformed':
      raise Exception('normal uninformed prior not yet implemented')
      # this isnt really the uninformed prior
      # prior_mean = 0
      # prior_var = 1
    return prior_mean, prior_var


  def __set_beta_posteriors(self, prior_distribution, raw_data_agg):
    """Set the pymc3 model to use Binomial posterior sampling."""
    bucketed_accts = raw_data_agg['bucketed'].values
    metric_agg = raw_data_agg[self.metric].values
    pm.Binomial('posterior', n=bucketed_accts, p=prior_distribution,
                observed=metric_agg) # posterior to do: observed may be a bug?


  def __set_poisson_posteriors(self, prior_distribution, raw_data_agg):
    """Set the pymc3 model to use Poisson posterior sampling."""
    metric_agg = raw_data_agg[self.metric].values
    pm.Poisson(name='posterior', mu=prior_distribution, observed=metric_agg)
    # to do: observed may be a bug?


  # http://www.ams.sunysb.edu/~zhu/ams570/Bayesian_Normal.pdf
  # https://www.mhnederlof.nl/bayesnormalupdate.html
  # https://en.wikipedia.org/wiki/Conjugate_prior
  def __norm_update_var(self, prior_var, sample):
    """Update variance to the posterior variance for a (log)
    normal distribution."""
    if self.prior_func == 'log-normal':
      variance = np.var(np.log(sample))
    elif self.prior_func == 'normal':
      variance = np.var(sample)
    posterior_var = ((1/prior_var) + (1/(variance/len(sample))))**(-1)
    return posterior_var


  def __norm_update_mean(self, posterior_var, prior_var,
                            prior_mean, sample):
    """Update mean to the posterior mean for a (log)normal distribution."""
    if self.prior_func == 'log-normal':
      mean = np.mean(np.log(sample))
      variance = np.var(np.log(sample))
    elif self.prior_func == 'normal':
      mean = np.mean(sample)
      variance = np.var(sample)
    posterior_mean = posterior_var*((prior_mean/prior_var) +
                                    (mean/(variance/len(sample))))
    return posterior_mean


  def __set_normal_posteriors(self, prior_mean, prior_var, samples):
    """Set the pymc3 model to use (log)normal posterior sampling.
    Posteriors are updated as a normal distribution.
    """
    posterior_variances = []
    posterior_means = []
    names = []

    for index, sample in enumerate(samples):
      posterior_var = self.__norm_update_var(prior_var, sample)
      posterior_mean = self.__norm_update_mean(posterior_var, prior_var,
                                                  prior_mean, sample)
      posterior_variances.append(posterior_var)
      posterior_means.append(posterior_mean)
      names.append(self.metric + str(index))

    if self.prior_func == 'log-normal':
      for var, mean, name in zip(posterior_variances, posterior_means, names):
        pm.Lognormal(name=name, mu=mean, sigma=np.sqrt(var), shape=1)
    elif self.prior_func == 'normal':
      for var, mean, name in zip(posterior_variances, posterior_means, names):
        pm.Normal(name=name, mu=mean, sigma=np.sqrt(var), shape=1)


  def __set_distributions(self):
    """Controller that calls correct prior and posterior functions based on
    the distribution called by instantiation.
    """
    if self.prior_func == 'beta':
      raw_data_agg = self.raw_data.groupby(self.bucket_col_name).sum()
      raw_data_agg['bucketed'] = self.raw_data.groupby(
                                  self.bucket_col_name).count()[self.metric]
      prior_distribution = self.__get_beta_priors(raw_data_agg)
      self.__set_beta_posteriors(prior_distribution, raw_data_agg)
    elif self.prior_func == 'poisson':
      raw_data_agg = self.raw_data.groupby(self.bucket_col_name).mean()
      raw_data_agg['variance'] = self.raw_data.groupby(
                                  self.bucket_col_name).var()[self.metric]
      raw_data_agg['bucketed'] = self.raw_data.groupby(
                                  self.bucket_col_name).count()[self.metric]
      prior_distribution = self.__get_poisson_priors(raw_data_agg)
      self.__set_poisson_posteriors(prior_distribution, raw_data_agg)
    elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
      C_data = self.raw_data.loc[self.raw_data[self.bucket_col_name] == self.control_bucket_name][self.metric].values
      samples = [list(C_data)]
      for variant in self.variant_bucket_names:
        samples.append(list(self.raw_data.loc[self.raw_data[self.bucket_col_name] == variant][self.metric].values))
      prior_mean, prior_var = self.__get_normal_priors(C_data)
      self.__set_normal_posteriors(prior_mean, prior_var, samples)


  def __set_samples(self):
    """Set private member variables to the correct sample."""
    if self.prior_func == 'beta' or self.prior_func == 'poisson':
      self.control_sample = self.__trace[self.metric][:,0]
      for i in range(1, 1+len(self.variant_bucket_names)):
        self.variant_samples.append(list(self.__trace[self.metric][:,i]))
    elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
      for index in range(0, len(self.variant_bucket_names) + 1):
        name = self.metric + str(index)
        if index == 0: self.control_sample = self.__trace[name][:,0]
        else: self.variant_samples.append(list(self.__trace[name][:,0]))

  #PUBLIC METHODS
  def create_report(self):
    """Utilize the private graphing functions to
    create a report of the test in one image.
    """
    if self.control_sample == []: raise Exception('create_model() must be' +
                                                  'run before create_report')

    fig = plt.figure(figsize=[12.8, 9.6])
    fig.suptitle('Bayesian AB Test Report for {}'.format(self.metric),
                                                         fontsize=16,
                                                         fontweight='bold')
    fig.subplots_adjust(hspace=1)

    # PDFS ONLY
    if not self.lift_plot_flag:
      self.__plot_posteriors()
      plt.show()

    # ONE VARIANT
    elif len(self.variant_bucket_names) == 1:
      plt.subplot(2,2,1)
      lift = self.lift[0]
      variant_name = self.variant_bucket_names[0]
      self.__plot_posteriors()
      plt.subplot(2,2,2)
      self.__plot_positive_lift(lift)
      plt.subplot(2,1,2)
      self.__plot_ecdf(variant_name)
      plt.show()

    # MULTIPLE VARIANTS
    else:
      nrows = len(self.variant_bucket_names) + 1
      if self.compare_variants:
        comparisons = list(range(0,len(self.variant_bucket_names)))
        combs = list(combinations(comparisons, 2))
        nrows += len(combs)
      plt.subplot(nrows,1,1)
      self.__plot_posteriors()
      loc = 3 # set initial figure location
      for num in range(0, len(self.variant_bucket_names)):
        plt.subplot(nrows,2,loc)
        self.__plot_positive_lift(self.lift[num], sample1=-1, sample2=num)
        plt.subplot(nrows,2,loc+1)
        self.__plot_ecdf(self.variant_bucket_names[num])
        loc += 2
      if self.compare_variants:
        for num in range(0, len(combs)):
          plt.subplot(nrows,2,loc)
          self.__plot_positive_lift(self.lift[num+2], sample1=combs[num][1],
                                    sample2=str(combs[num][0]))
          plt.subplot(nrows,2,loc+1)
          name = 'bucket_comparison' + \
                  str(combs[num][1]) + '_' + str(combs[num][0])
          self.__plot_ecdf(name)
          loc += 2
      plt.show()

  def plot_posteriors(self):
    """Public interface to plot just the posteriors."""
    # to do: add ability to call with variable names
    self.__plot_posteriors()
    plt.show()

  # to do: implement
  # def plot_positive_lift(self, ):
  #   self.__plot_positive_lift()

  # to do: implement
  # def plot_ecdf(self):
  #   self.__plot_ecdf()


  def create_model(self):
    """Set up the pymc3 model with the prior parameters."""
    model = pm.Model()
    with model:
      self.__set_distributions()

    if self.debug:
      print('Running', self.samples, 'MCMC simulations')
    with model:
      self.__trace = pm.sample(self.samples)
      self.__set_samples()

    if self.debug:
      summary = pm.summary(self.__trace)
      print('Posterior sampling distribution summary statistics' +
            '\n {}'.format(summary[['mean', 'sd']].round(4)))

    if (self.lift_plot_flag) or (not self.lift_plot_flag and self.debug):
      self.__calc_lift()
      self.__calc_ecdf()

  def loss_function(self, loss_type='absolute', epsilon=0.0001):
    """Calculate the loss function for the variants.
    Currently only conversion 1-variant is supported."""
    # REFERENCE: https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html
    # https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf

    # "This stopping condition considers both the likelihood that β — α
    # is greater than zero and also the magnitude of this difference.
    # https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5

    # currently only analytical solution for a one variant beta test is implemeneted
    # will have to numerically do other ones

    if loss_type not in ['absolute','percent']:
      raise Exception('loss_type must be either absolute or percent')
    if self.prior_func != 'beta':
      raise Exception('prior_func must be beta')
    if len(self.variant_bucket_names) != 1:
      raise Exception('only one variant is supported')

    raw_data_agg = self.raw_data.groupby(self.bucket_col_name).sum()
    raw_data_agg['bucketed'] = self.raw_data.groupby(
                                      self.bucket_col_name).count()[self.metric]

    # to do: this is not very clean
    if self.prior_info == 'uninformed':
      prior_alpha = 1
      prior_beta = 1
    else:
      prior_converts = raw_data_agg.loc[self.control_bucket_name][self.metric]
      prior_non_converts = raw_data_agg.loc[self.control_bucket_name] \
                                           ['bucketed'] - prior_converts
      prior_alpha = prior_converts / self.prior_scale_factor
      prior_beta = prior_non_converts / self.prior_scale_factor

    control_alpha_likelihood = raw_data_agg.loc[self.control_bucket_name][self.metric]
    control_beta_likelihood = raw_data_agg.loc[self.control_bucket_name]['bucketed'] - control_alpha_likelihood

    variant_alpha_likelihood = raw_data_agg.loc[self.variant_bucket_names[0]][self.metric]
    variant_beta_likelihood = raw_data_agg.loc[self.variant_bucket_names[0]]['bucketed'] - variant_alpha_likelihood

    a = control_alpha_likelihood + prior_alpha
    b = control_beta_likelihood + prior_beta

    c = variant_alpha_likelihood + prior_alpha
    d = variant_beta_likelihood + prior_beta

    A_risk = self.__loss_beta(c, d, a, b, loss_type=loss_type)
    B_risk = self.__loss_beta(a, b, c, d, loss_type=loss_type)
    stop, winner = self.__stop_test(A_risk, B_risk, epsilon=epsilon)

    if stop: print('Test can be stopped and', winner, 'can be declared the winner')
    else: print('Keep running the test')
    print('risk of choosing', self.control_bucket_name, '=', A_risk)
    print('risk of choosing', self.variant_bucket_names[0], '=', B_risk)
