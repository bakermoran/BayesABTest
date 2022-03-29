"""Plotting functionality for ab_test_model."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations
from ._ab_test_model_utils import _ab_test_utils
# pylint: disable=no-member


class _ab_test_plotting(_ab_test_utils):
    """Provide Funcs for class to plot Bayesian charts."""

    def _plot_posteriors(self, variants=[]):
        """Plot KDE of the posterior samples.

        This is a private function. For a public interface, see
            plot_posteriors().

        Keyword Arguments:
            variants {list} -- which variants to plot. If empty, all are
                plotted. Otherwise, the must be contained in raw_data
                (default: {[]}).
        """
        if variants == []:
            variants = list(self.posteriors.keys())
        for variant in variants:
            sns.kdeplot(self.posteriors[variant].get_posterior_sample(),
                        shade=True,
                        color=self.posteriors[variant].get_color())
        plt.legend(labels=variants, loc='upper right')
        if self.prior_function == 'beta':
            plt.xlabel('Conversion Rate')
        elif (self.prior_function == 'log-normal'
              or self.prior_function == 'normal'):
            plt.xlabel(self.metric)
        sns.despine(left=True)
        plt.yticks([], [])
        title = 'Distribution(s) for {0} for {1}'.format(
                                            self._stringify_variants(variants),
                                            self.metric)
        title = self._format_title(title)
        plt.title(title)
        if self.prior_function == 'beta':
            locs, labels = plt.xticks()
            labels = self._format_axis_as_percent(locs, labels)
            plt.xticks(locs, labels=labels)

    def _plot_positive_lift(self, numerator_name, denominator_name):
        """Plot the lift vector as a kernel density estimation.

        This is a private function. For a public interface, see
            plot_positive_lift().

        Arguments:
            numerator_name {str} -- The name of the numerator in the lift
                calculation.
            denominator_name {str} -- The name of the numerator in the lift
                calculation.
        """
        lift = self.lift[numerator_name][denominator_name]
        ax = sns.kdeplot(lift, shade=True)
        line = ax.get_lines()[0]
        x, y = line.get_data()
        mask = x > 0
        x, y = x[mask], y[mask]
        ax.fill_between(x, y1=y, alpha=0.5, facecolor='red')
        if len(self.variant_bucket_names) > 1:
            title = numerator_name + ' vs ' + denominator_name
            ax.set_ylabel(title, rotation=0, fontstyle='italic')
        plt.axvline(x=0, linestyle='dotted', color='black')
        plt.xlabel('Lift')
        percent_positive_lift = sum(i > 0 for i in lift) / len(lift)
        title = '{0} had {1:.2%} probability of positive lift'.format(
                                                        self.metric,
                                                        percent_positive_lift)
        title = self._format_title(title)
        plt.title(title)
        sns.despine(left=True)
        plt.yticks([], [])
        locs, labels = plt.xticks()
        labels = self._format_axis_as_percent(locs, labels)
        plt.xticks(locs, labels=labels)

    def _plot_ecdf(self, numerator_name, denominator_name):
        """Plot the empirical cumulative distribution function.

        This is a private function. For a public interface, see
            plot_ecdf().

        Arguments:
            numerator_name {str} -- The name of the numerator in the lift
                calculation.
            denominator_name {str} -- The name of the numerator in the lift
                calculation.
        """
        x = self.ecdf[numerator_name][denominator_name]['x']
        y = self.ecdf[numerator_name][denominator_name]['y']

        lower_bound = x[y.index(min(y,
                                    key=lambda x:
                                    abs(x - self.confidence_level / 2)))]
        median = x[y.index(min(y, key=lambda x:abs(x - 0.5)))]
        upper_bound = x[y.index(min(y,
                                    key=lambda x:
                                    abs(x - (1 - self.confidence_level / 2))))]

        sns.lineplot(x=x, y=y)
        ci = 1 - self.confidence_level
        title = ('Median Lift was {0:.2%}, with a '
                 '{1:.0%} CI of [{2:.2%}, {3:.2%}]'.format(median,
                                                           ci,
                                                           lower_bound,
                                                           upper_bound))
        title = self._format_title(title)
        plt.title(title)
        plt.xlabel('Lift')
        plt.ylabel('Cumulative Probability')
        plt.axvline(x=lower_bound, linestyle='dotted', color='black')
        plt.axvline(x=median, linestyle='dotted', color='black')
        plt.axvline(x=upper_bound, linestyle='dotted', color='black')
        sns.despine(left=True)
        locs, labels = plt.xticks()
        labels = self._format_axis_as_percent(locs, labels)
        plt.xticks(locs, labels=labels)

    def _calc_ecdf(self):
        """Calculate the empirical CDFs and set member var."""
        for numerator, vals in self.lift.items():
            for denominator, lift in vals.items():
                raw_data = np.array(lift)
                cdfx = np.sort(np.unique(lift))
                x_values = np.linspace(start=min(cdfx),
                                       stop=max(cdfx),
                                       num=len(cdfx))
                size_data = raw_data.size
                y_values = []
                for i in x_values:
                    temp = raw_data[raw_data <= i]
                    value = temp.size / size_data
                    y_values.append(value)
                temp = {}
                temp['x'] = x_values
                temp['y'] = y_values
                if numerator not in self.ecdf.keys():
                    self.ecdf[numerator] = {}
                    self.ecdf[numerator][denominator] = temp
                else:
                    self.ecdf[numerator][denominator] = temp

    def _calc_lift(self):
        """Calculate the lift of the variants over the others."""
        for key, val in self.posteriors.items():
            if key == self.control_bucket_name:
                continue
            lift_over_control = np.divide(val.get_posterior_sample(),
                                          self.posteriors[
                                           self.control_bucket_name]
                                          .get_posterior_sample()) - 1
            if key not in self.lift.keys():
                self.lift[key] = {}
                self.lift[key][self.control_bucket_name] = lift_over_control
            else:
                self.lift[key][self.control_bucket_name] = lift_over_control
            if self.debug:
                percent_positive_lift = sum(i > 0 for i in
                                            lift_over_control) / \
                                            len(lift_over_control)
                print('percent positive lift for {0} over {1} = {2:.2%}'
                      .format(key, self.control_bucket_name,
                              percent_positive_lift))

        if self.compare_variants:
            comparisons = list(range(0, len(self.variant_bucket_names)))
            combs = combinations(comparisons, 2)
            for combination in combs:
                denom = self.posteriors[
                            self.variant_bucket_names[combination[0]]]
                num = self.posteriors[
                            self.variant_bucket_names[combination[1]]]
                lift = np.divide(num.get_posterior_sample(),
                                 denom.get_posterior_sample()) - 1
                if num.get_variant_name() not in self.lift.keys():
                    self.lift[num.get_variant_name()] = {}
                    self.lift[num.get_variant_name()][
                        denom.get_variant_name()] = lift
                else:
                    self.lift[num.get_variant_name()][
                        denom.get_variant_name()] = lift
                if self.debug:
                    percent_positive_lift = sum(i > 0 for i in lift) \
                                                / len(lift)
                    print('percent positive lift for {0} over {1} = {2:.2%}'
                          .format(num.get_variant_name(),
                                  denom.get_variant_name(),
                                  percent_positive_lift))

    def plot_posteriors(self, variants=[]):
        """Plot the PDFs of the posterior distributions.

        Arguments:
            variants {list} -- List of variant names to be plotted.
            If variants is not set, all are plotted, otherwise, the variants
            in the list are plotted. Variants must only have items in
            bucket_col_name (default: {[]}).
        """
        if variants != []:
            for var in variants:
                if var not in self.posteriors.keys():
                    raise ValueError(('Variants must only be a value in '
                                      'bucket_col_name'))
        self._plot_posteriors(variants)

    def plot_positive_lift(self, variant_one, variant_two):
        """Plot the positive lift pdt between variant_one and variant_two.

        Arguments:
        variant_one and variant_two should not be the same
            variant_one {str} -- should be a value in bucket_col_name.
            variant_two {str} -- should be a value in bucket_col_name.
        """
        if variant_one == variant_two:
            raise ValueError('variant_one and variant_two cannot be the same')
        if variant_one not in self.posteriors.keys() or \
                variant_two not in self.posteriors.keys():
            raise ValueError(('Variants must only be a value in column '
                              '{}'.format(self.bucket_col_name)))

        if variant_one != self.control_bucket_name and \
                variant_two != self.control_bucket_name:
            if not self.compare_variants:
                raise RuntimeError('Compare_variants must be set to true in '
                                   'order to compare {0} and {1}'
                                   .format(variant_one, variant_two))
        if variant_one in self.lift.keys() and \
                variant_two in self.lift[variant_one].keys():
            self._plot_positive_lift(numerator_name=variant_one,
                                     denominator_name=variant_two)
        else:
            self._plot_positive_lift(numerator_name=variant_two,
                                     denominator_name=variant_one)

    def plot_ecdf(self, variant_one, variant_two):
        """Plot the empirical cdf for the lift b/w variant_one and variant_two.

        Arguments:
            variant_one {str} -- should be a value in bucket_col_name.
            variant_two {str} -- should be a value in bucket_col_name.
        """
        if variant_one == variant_two:
            raise ValueError('variant_one and variant_two cannot be the same')
        if variant_one not in self.posteriors.keys() or \
                variant_two not in self.posteriors.keys():
            raise ValueError(('Variants must only be a value in column '
                              '{}'.format(self.bucket_col_name)))

        if variant_one in self.ecdf.keys() and \
                variant_two in self.ecdf[variant_one].keys():
            self._plot_ecdf(numerator_name=variant_one,
                            denominator_name=variant_two)
            plt.ylabel('Cumulative Lift: {0} vs {1}'
                       .format(variant_two, variant_one))
        else:
            self._plot_ecdf(numerator_name=variant_two,
                            denominator_name=variant_one)
            plt.ylabel('Cumulative Lift: {0} vs {1}'
                       .format(variant_one, variant_two))
