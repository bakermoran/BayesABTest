"""Declaration file for ab_test_model class."""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from ._ab_test_model_plotting import _ab_test_plotting
from ._ab_test_model_sampling import _ab_test_model_sampling
from ._ab_test_model_loss_func import _ab_test_loss_functions
from ._prior_distribution import _prior_distribution_params
from ._posterior_distribution import posterior_distribution

# empirical bayes method https://en.wikipedia.org/wiki/Empirical_Bayes_method
# https://juanitorduz.github.io/intro_pymc3/
# https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.Gamma
# https://docs.pymc.io/api/distributions/discrete.html#pymc3.distributions.discrete.Poisson


class ab_test_model(_ab_test_plotting,
                    _ab_test_model_sampling,
                    _ab_test_loss_functions):
    """Fit an ab_test_model object to your data, and output the results.

    This is a easy to use wrapper around scipy and numpy, with built in
    plotting and reporting.

    [methods]
        fit()
            requires -- class was instantiated with valid inputs
            modifies -- ecdf, control_sample, variant_sample, lift
            effects --  creates and runs the monte carlo simulation, sets
                        member variables to reflect model outputs

        plot()
            requires -- fit() has been run
            modifies -- none
            effects -- creates a 3 chart report of the AB test and opens it
                       in a python viewer

    [runtime variables]
        ecdf -- a dictionary with keys x and y, and the value is a list
                containing the x and y coordinates for the empirical CDF
                (only meaningful after fit() is run)
        control_sample -- a list containing the sampled values from the
                            control posterior distribution (only meaningful
                            after fit() is run)
        variant_sample -- a list containing the sampled values from the
                            variant posterior distribution (only meaningful
                            after fit() is run)
        lift -- a list containing the lift of the variant over the control
                for each sampled value (only meaningful after fit() is run)
    """

    def __init__(self, raw_data, metric, prior_function, samples=10000,
                 prior_info='informed', control_bucket_name='off',
                 bucket_col_name='bucket', confidence_level=.05,
                 compare_variants=False, prior_scale_factor=4,
                 prior_parameters=None, debug=False):
        """
        Initialize an instance of BayesABTest with input variables.

        Arguments:
            raw_data {pandas DF} -- a pandas dataframe with (at least)
                two columns, one for the bucket and one for
                the response variable
            metric {str} -- column name in raw_data for the
                response variable

        Keyword Arguments:
            samples {int} -- number of samples to run the monte carlo
                simulation, must be 1,000 and 50,000 (default: {10000})
            prior_function {str} --the type of distribution to use for
                the prior.
                (default: {'beta'}). options are:
                beta -- use for conversion rates. bounded on the interval [0,1]
                log--normal -- use for continuous, greater than zero response
                    variables (0, +inf) (ex: premium, minutes
                    spent on scene, etc.)
                normal -- use for continuous response variables that on
                    the interval (-inf, +inf)
                poisson -- use for discrete, greater than zero response
                    variables (ex: arrivals per day, events per
                    account, etc.)
            prior_info {str} -- the prior belief to about the response varable.
                (default: {'informed'})
                options are:
                uninformed -- no prior belief about the response, all outcomes
                    are equaly likely
                informed -- uses the control as the prior belief, will add
                    support to scale down that belief to a weaker belief
                specified -- user specified prior parameters

            control_bucket_name {str} -- value in bucket_col_name for the
                control bucket (default: {'off'})
            bucket_col_name {str} -- column name in raw_data for the
                bucket  (default: {'bucket'})
            confidence_level {float} -- value for the confidence interval
                on the CDF chart (defaults to 0.05) (default: {.05})
            compare_variants {bool} -- boolean for comparing the variants to
                each other. Control to each variant is always done (unless
                there are too many variants to plot). If there
                are few enough variants, the comparisons for
                variants will be plotted. (default: {False})
            prior_scale_factor {int} -- actor to scale an informed prior by.
                ignored if prior params are user specified (default: {4})
            prior_parameters {[type]} -- [description] (default: {None})
            debug {bool} -- [boolean to print out extra output for debugging
            purposes (default: {False})
        """
        buckets = raw_data[bucket_col_name].unique()
        self._error_check_inputs(raw_data, metric, samples, prior_function,
                                 prior_info, control_bucket_name, buckets,
                                 bucket_col_name, confidence_level,
                                 compare_variants, prior_scale_factor,
                                 prior_parameters, debug)

        self.posteriors = {}
        self.control_bucket_name = control_bucket_name
        self.variant_bucket_names = []

        colors = ['r', 'g', 'y', 'm', 'c', 'darkred', 'darkorange',
                  'indigo', 'slategrey', 'coral']
        color_index = 0
        for bucket in buckets:
            if bucket == control_bucket_name:
                self.posteriors[bucket] = posterior_distribution(
                    bucket,
                    'blue',
                    prior_function)
            else:
                self.posteriors[bucket] = posterior_distribution(
                                                bucket, colors[color_index],
                                                prior_function)
                color_index += 1
        self.variant_bucket_names = list(self.posteriors.keys())
        self.variant_bucket_names.remove(self.control_bucket_name)

        # PUBLIC VARIABLES
        self.debug = debug
        self.prior_info = prior_info
        self.prior_function = prior_function
        self.raw_data = raw_data
        self.metric = metric
        self.samples = samples
        self.compare_variants = compare_variants
        self.bucket_col_name = bucket_col_name
        self.prior_parameters = prior_parameters
        self.confidence_level = confidence_level
        self.prior_scale_factor = prior_scale_factor

        if samples * (1 + len(self.variant_bucket_names)) > 50000:
            print('WARNING: This is a large amount of sampling.',
                  'Run time may be long.')
            print('Running', samples, 'on', 1 + len(self.variant_bucket_names),
                  'PDFs, for a total of',
                  samples * (1 + len(self.variant_bucket_names)),
                  'samples')
        self.ecdf = {}
        self.lift = {}

        self.prior_params = _prior_distribution_params(self)
        self._trace = {}

    def _error_check_inputs(self, raw_data, metric, samples, prior_function,
                            prior_info, control_bucket_name, buckets,
                            bucket_col_name, confidence_level,
                            compare_variants, prior_scale_factor,
                            prior_parameters, debug):
        """Check the input variables for errors.

        This class only supports the inputs in __init__.
        """
        # to do: naming is wrong cuz some are priors some are posts
        SUPPORTED_PRIOR_FUNCTIONS = ['beta', 'log-normal', 'normal', 'poisson']
        SUPPORTED_PRIOR_INFO_TYPES = ['informed', 'uninformed', 'specified']

        if raw_data.empty:
            raise ValueError('Input dataframe must contain data')
        if prior_function == 'beta' \
                and ([np.any(x) for x in raw_data[metric].unique()
                     if x not in list([0, 1])] or [False])[0]:
            raise ValueError('Observed data for beta prior should be binary')
        if metric not in raw_data.columns:
            raise ValueError('Input dataframe must contain column:', metric)
        if samples < 1000 or samples > 50000:
            raise ValueError(('Number of samples must be in the '
                              'interval [1000,50000]'))
        if prior_function not in SUPPORTED_PRIOR_FUNCTIONS:
            raise ValueError('Prior must be' +
                             'in [{}]'.format(
                                 ', '.join(SUPPORTED_PRIOR_FUNCTIONS)))
        if prior_info not in SUPPORTED_PRIOR_INFO_TYPES:
            raise ValueError('Prior must be in' +
                             '[{}]'.format(
                                 ', '.join(SUPPORTED_PRIOR_INFO_TYPES)))
        if prior_info == 'specified' and prior_parameters is None:
            raise ValueError(('If prior_info == specifed, '
                             'prior_parameters must not be None'))
        if prior_info != 'specified' and prior_parameters is not None:
            print(('WARNING: prior_info was specified as {}. prior_parameters '
                  'are being ignored'.format(prior_parameters)))
        if control_bucket_name not in raw_data[bucket_col_name].unique():
            raise ValueError('Input dataframe', bucket_col_name,
                             'column must contain values with:',
                             control_bucket_name,
                             'as the control bucket name')
        if compare_variants and len(buckets) - 1 < 2:
            raise ValueError(('If compare_variants is True, there must be '
                             'at least 2 variants'))
        if len(buckets) - 1 > 10:
            raise RuntimeError(('Greater than 10 variants is not currently '
                               'supported'))
        if bucket_col_name not in raw_data.columns:
            raise ValueError('Input dataframe must contain column:',
                             bucket_col_name)
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError('Confidence level must be in the interval (0,1)')
        if debug not in [True, False]:
            raise ValueError('debug must be either True or False')
        if compare_variants not in [True, False]:
            raise ValueError('compare_variants must be either True or False')

    def fit(self):
        """Set up the model, run sampling, and calculate lift."""
        print('Running', self.samples * (1 + len(self.variant_bucket_names)),
              'MCMC simulations')

        self._sample()

        self._calc_lift()
        self._calc_ecdf()

    def plot(self, lift_plot_flag=True):
        """Utilize the private graphing functions to create a report.

        Arguments:
            lift_plot_flag -- boolean for plotting lift PDF and
            CDF (default: {True})
        """
        if self.posteriors[self.control_bucket_name].get_posterior_sample() \
                is None:
            raise UnboundLocalError('fit() must be run before plot')

        if lift_plot_flag and len(self.variant_bucket_names) > 3 \
                and self.compare_variants:
            lift_plot_flag = False
            print(('WARNING: lift_plot_flag reset to False due to the large '
                   'number of combinations of variant comparisons.'))
            print(('Re-run with less variants selected for comparison, or '
                   'turn off compare_variants'))
        if len(self.variant_bucket_names) > 5:
            lift_plot_flag = False
            print(('WARNING: lift PDF and CDF graphing is turned off for '
                   'more than 5 varaints. Only the posterior PDFs will '
                   'be plotted.'))
            print(('Re-run with less variants selected for lift comparisons, ',
                   'or run with debug=True to see relative lifts in ',
                   'command line.'))

        fig = plt.figure(figsize=[12.8, 9.6])
        fig.suptitle('Bayesian AB Test Report for {}'.format(self.metric),
                     fontsize=16,
                     fontweight='bold')
        fig.subplots_adjust(hspace=1)

        # PDFS ONLY
        if not lift_plot_flag:
            self._plot_posteriors()

        # ONE VARIANT
        # this is separate because the layout of the report is different
        elif len(self.variant_bucket_names) == 1:
            plt.subplot(2, 2, 1)
            variant_name = self.variant_bucket_names[0]
            self._plot_posteriors()
            plt.subplot(2, 2, 2)
            self._plot_positive_lift(numerator_name=variant_name,
                                     denominator_name=self.control_bucket_name)
            plt.subplot(2, 1, 2)
            self._plot_ecdf(numerator_name=variant_name,
                            denominator_name=self.control_bucket_name)

        # MULTIPLE VARIANTS
        else:
            nrows = len(self.variant_bucket_names) + 1
            if self.compare_variants:
                comparisons = list(range(0, len(self.variant_bucket_names)))
                combs = list(combinations(comparisons, 2))
                nrows += len(combs)
            plt.subplot(nrows, 1, 1)
            self._plot_posteriors()
            loc = 3
            for numerator, val in self.lift.items():
                for denominator in val.keys():
                    if denominator != self.control_bucket_name:
                        continue
                    plt.subplot(nrows, 2, loc)
                    self._plot_positive_lift(numerator_name=numerator,
                                             denominator_name=denominator)
                    plt.subplot(nrows, 2, loc+1)
                    self._plot_ecdf(numerator_name=numerator,
                                    denominator_name=denominator)
                    loc += 2
            if self.compare_variants:
                for numerator, val in self.lift.items():
                    for denominator in val.keys():
                        if denominator == self.control_bucket_name:
                            continue
                        plt.subplot(nrows, 2, loc)
                        self._plot_positive_lift(numerator_name=numerator,
                                                 denominator_name=denominator)
                        plt.subplot(nrows, 2, loc+1)
                        self._plot_ecdf(numerator_name=numerator,
                                        denominator_name=denominator)
                        loc += 2
