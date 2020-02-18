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
        colors = ['blue', 'red', 'green', 'yellow', 'purple', 'cyan']
        colors += colors
        legend_labels = []
        if variants == [] or self.control_bucket_name in variants:
            sns.kdeplot(self.control_sample, shade=True, color=colors[0])
            legend_labels.append(self.control_bucket_name)
        for i in range(0, len(self.variant_bucket_names)):
            if variants == [] or self.variant_bucket_names[i] in variants:
                sns.kdeplot(self.variant_samples[i],
                            shade=True,
                            color=colors[i+1])
                legend_labels.append(self.variant_bucket_names[i])
        plt.legend(labels=legend_labels, loc='upper right')
        if self.prior_func == 'beta':
            plt.xlabel('Conversion Rate')
        elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
            plt.xlabel(self.metric)
        sns.despine(left=True)
        plt.yticks([], [])
        title = 'Distribution(s) for {0} for {1}'.format(
                                            self._stringify_variants(variants),
                                            self.metric)
        title = self._format_title(title)
        plt.title(title)
        if self.prior_func == 'beta':
            locs, labels = plt.xticks()
            labels = self._format_axis_as_percent(locs, labels)
            plt.xticks(locs, labels=labels)

    def _plot_positive_lift(self, lift, sample1=None, sample2=None):
        """Plot the lift vector as a kernel density estimation.

        This is a private function. For a public interface, see
            plot_positive_lift().

        Arguments:
            lift {list} -- The lift vector to be plotted

        Keyword Arguments:
            sample1 {int} -- position in variant names vector.
                This is the denominator. -1 for sample1 means it is the
                control. (default: {None})
            sample2 {int} -- position in variant names vector.
                This is the numerator. (default: {None})
        """
        ax = sns.kdeplot(lift, shade=True)
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
            else:
                denom_name = self.variant_bucket_names[sample1]
            num_name = self.variant_bucket_names[sample2]
            title = num_name + ' vs ' + denom_name
            ax.set_ylabel(title, rotation=0, fontstyle='italic')
        plt.vlines(0.0, ymin=0, ymax=ylim, linestyle='dotted')
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

    def _plot_ecdf(self, variant_name):
        """Plot the empirical cumulative distribution function.

        This is a private function. For a public interface, see
            plot_ecdf().

        Arguments:
            variant_name {str} -- the string of the variant combo to be
                plotted. This is purely an internal convention, and is not
                meant to be exposed. This is generated within the model.
        """
        x = self.ecdf[variant_name]['x']
        y = self.ecdf[variant_name]['y']

        lower_bound = x[y.index(min(y,
                                    key=lambda x:
                                    abs(x-self.confidence_level)))]
        median = x[y.index(min(y, key=lambda x:abs(x-0.5)))]
        upper_bound = x[y.index(min(y,
                                    key=lambda x:
                                    abs(x-(1-self.confidence_level))))]

        sns.lineplot(x=self.ecdf[variant_name]['x'],
                     y=self.ecdf[variant_name]['y'])
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
        plt.vlines(lower_bound, ymin=0, ymax=1, linestyle='dotted')
        plt.vlines(median, ymin=0, ymax=1, linestyle='dotted')
        plt.vlines(upper_bound, ymin=0, ymax=1, linestyle='dotted')
        sns.despine(left=True)
        locs, labels = plt.xticks()
        labels = self._format_axis_as_percent(locs, labels)
        plt.xticks(locs, labels=labels)

    def _calc_ecdf(self):
        """Calculate the empirical CDFs and set member var."""
        if self.compare_variants:
            comparisons = list(range(0, len(self.variant_bucket_names)))
            combs = list(combinations(comparisons, 2))
        num_variants = len(self.variant_bucket_names)
        for index, lift in enumerate(self.lift):
            raw_data = np.array(lift)
            cdfx = np.sort(np.unique(self.lift))
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
            if index < num_variants:
                self.ecdf[self.variant_bucket_names[index]] = temp
            elif self.compare_variants and index >= num_variants:
                name = 'bucket_comparison' + \
                            str(combs[index-num_variants][1]) + \
                            '_' + str(combs[index-num_variants][0])
                self.ecdf[name] = temp

    def _calc_lift(self):
        """Calculate the lift of the variants over the control samples."""
        for index, sample in enumerate(self.variant_samples):
            self.lift.append(sample / self.control_sample - 1)
            if self.debug:
                percent_positive_lift = sum(i > 0 for i in
                                            self.lift[index]) / \
                                            len(self.lift[index])
                print('percent positive lift for {0} over {1} = {2:.2%}'
                      .format(self.variant_bucket_names[index],
                              self.control_bucket_name,
                              percent_positive_lift))

        if self.compare_variants:
            comparisons = list(range(0, len(self.variant_bucket_names)))
            combs = combinations(comparisons, 2)
            for combination in combs:
                self.lift.append(np
                                 .array(self.variant_samples[combination[1]])
                                 / np.array(self
                                            .variant_samples[
                                                   combination[0]]) - 1)
                if self.debug:
                    percent_positive_lift = sum(i > 0 for i in self.lift[-1]) \
                                                / len(self.lift[-1])
                    print('percent positive lift for {0} over {1} = {2:.2%}'
                          .format(self.variant_bucket_names[combination[1]],
                                  self.variant_bucket_names[combination[0]],
                                  percent_positive_lift))

    def plot_posteriors(self, variants=[]):
        """Plot the PDFs of the posterior distributions.

        Arguments:
            variants {list} -- List of variant names to be plotted.
            If variants is not set, all are plotted, otherwise, the variants
            in the list are plotted. Variants must only have items in
            control_bucket_name, or variant_bucket_names (default: {[]}).
        """
        for var in variants:
            if var not in self.variant_bucket_names \
                    and var != self.control_bucket_name:
                raise Exception(('variants must only have items in '
                                 'control_bucket_name, or '
                                 'variant_bucket_names'))
        self._plot_posteriors(variants)

    def plot_positive_lift(self, variant_one, variant_two):
        """Plot the positive lift pdt between variant_one and variant_two.

        Arguments:
        variant_one and variant_two should not be the same
            variant_one {str} -- should be either one of control_bucket_name
                or in variant_bucket_names.
            variant_two {str} -- should be either one of control_bucket_name
                or in variant_bucket_names.
        """
        if variant_one == variant_two:
            raise Exception('variant_one and variant_two cannot be the same')
        if variant_one != self.control_bucket_name and \
                variant_one not in self.variant_bucket_names:
            raise Exception('variant_one must be one of {0}, or {1}'
                            .format(self.control_bucket_name,
                                    self.variant_bucket_names))
        if variant_one != self.control_bucket_name and \
                variant_one not in self.variant_bucket_names:
            raise Exception('variant_one must be one of {0}, or {1}'
                            .format(self.control_bucket_name,
                                    self.variant_bucket_names))

        if variant_one == self.control_bucket_name or \
                variant_two == self.control_bucket_name:
            sample1 = -1
            if variant_two != self.control_bucket_name:
                sample2 = self.variant_bucket_names.index(variant_two)
            else:
                sample2 = self.variant_bucket_names.index(variant_one)
        else:
            if not self.compare_variants:
                # to do: this is dumb, shouldnt be a requirement
                raise Exception("""compare_variants must be set to true in
                                order to compare {0} and {1}"""
                                .format(variant_one, variant_two))
            sample1 = min(self.variant_bucket_names.index(variant_one),
                          self.variant_bucket_names.index(variant_two))
            sample2 = max(self.variant_bucket_names.index(variant_one),
                          self.variant_bucket_names.index(variant_two))

        if sample1 == -1:
            self._plot_positive_lift(self.lift[sample2],
                                     sample1=-1,
                                     sample2=sample2)
        else:
            self._plot_positive_lift(self.lift[
                                     sample2+len(self.variant_bucket_names)-1],
                                     sample1=sample1,
                                     sample2=sample2)

    def plot_ecdf(self, variant_one, variant_two):
        """Plot the empirical cdf for the lift b/w variant_one and variant_two.

        Arguments:
            variant_one {str} -- should be either one of control_bucket_name
                or in variant_bucket_names.
            variant_two {str} -- should be either one of control_bucket_name
                or in variant_bucket_names.
        """
        if variant_one == variant_two:
            raise Exception('variant_one and variant_two cannot be the same')
        if variant_one != self.control_bucket_name and \
                variant_one not in self.variant_bucket_names:
            raise Exception('variant_one must be one of {0}, or {1}'
                            .format(self.control_bucket_name,
                                    self.variant_bucket_names))
        if variant_one != self.control_bucket_name and \
                variant_one not in self.variant_bucket_names:
            raise Exception('variant_one must be one of {0}, or {1}'
                            .format(self.control_bucket_name,
                                    self.variant_bucket_names))

        if variant_one == self.control_bucket_name:
            self._plot_ecdf(variant_two)
            plt.ylabel('Cumulative Lift: {0} vs {1}'
                       .format(variant_two, variant_one))
        elif variant_two == self.control_bucket_name:
            self._plot_ecdf(variant_one)
            plt.ylabel('Cumulative Lift: {0} vs {1}'
                       .format(variant_one, variant_two))
        elif self.variant_bucket_names.index(variant_one) > \
                self.variant_bucket_names.index(variant_two):
            name = 'bucket_comparison' + \
                    str(self.variant_bucket_names.index(variant_one)) + \
                    '_' + str(self.variant_bucket_names.index(variant_two))
            self._plot_ecdf(name)
            plt.ylabel('Cumulative Lift: {0} vs {1}'
                       .format(variant_one, variant_two))
        else:
            name = 'bucket_comparison' + \
                    str(self.variant_bucket_names.index(variant_two)) + \
                    '_' + str(self.variant_bucket_names.index(variant_one))
            self._plot_ecdf(name)
            plt.ylabel('Cumulative Lift: {0} vs {1}'
                       .format(variant_two, variant_one))
