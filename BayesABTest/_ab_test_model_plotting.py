"""Plotting functionality for ab_test_model."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations
from ._ab_test_model_utils import _ab_test_utils
#pylint: disable=no-member

class _ab_test_plotting(_ab_test_utils):
  def _plot_posteriors(self):
    """Plot kernel density estimations of the posterior samples."""
    colors = ['blue', 'red', 'green', 'yellow', 'purple', 'cyan']
    colors += colors
    sns.kdeplot(self.control_sample, shade=True, color=colors[0])
    for i in range(0, len(self.variant_bucket_names)):
      sns.kdeplot(self.variant_samples[i], shade=True, color=colors[i+1])
    plt.legend(labels=self._buckets, loc='upper right')
    if self.prior_func == 'beta': plt.xlabel('Conversion Rate')
    elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
      plt.xlabel(self.metric)
    sns.despine(left=True)
    plt.yticks([],[])
    title = 'Distributions for {0} for {1}'.format(self._stringify_variants(),
                                                    self.metric)
    title = self._format_title(title)
    plt.title(title)
    if self.prior_func == 'beta':
      locs, labels = plt.xticks()
      labels = self._format_axis_as_percent(locs, labels)
      plt.xticks(locs, labels=labels)


  def _plot_positive_lift(self, lift, sample1=None, sample2=None):
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
    title = self._format_title(title)
    plt.title(title)
    sns.despine(left=True)
    plt.yticks([],[])
    locs, labels = plt.xticks()
    labels = self._format_axis_as_percent(locs, labels)
    plt.xticks(locs, labels=labels)


  def _plot_ecdf(self, variant_name):
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
    title = self._format_title(title)
    plt.title(title)
    plt.xlabel('Lift')
    plt.ylabel('Cumulative Probability')
    plt.vlines(lower_bound,ymin=0,ymax=1,linestyle='dotted')
    plt.vlines(median,ymin=0,ymax=1,linestyle='dotted')
    plt.vlines(upper_bound,ymin=0,ymax=1,linestyle='dotted')
    sns.despine(left=True)
    locs, labels = plt.xticks()
    labels = self._format_axis_as_percent(locs, labels)
    plt.xticks(locs, labels=labels)


  def _calc_ecdf(self):
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


  def _calc_lift(self):
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
