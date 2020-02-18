"""Class declaration file for class _prior_distribution_params."""
import numpy as np


class _prior_distribution_params:
    """Handle all of the prior information needed for the AB test model.

    Inputs are inputted from the ab_test_model class.
    """

    def __init__(self, ab_test):
        """Set prior distribution parameters an AB test.

        Arguments:
            ab_test {ab_test_model} -- an valid, instantiated ab_test_model
                object to set prior parameters for.
        """
        if ab_test.prior_info == 'specified' and \
                type(ab_test.prior_parameters) != dict:
            raise Exception('prior_params must be dict type')

        self.ab_test = ab_test

        self.alpha = None
        self.beta = None

        self.mean = None
        self.var = None

        self.raw_data_agg = None
        self.control_data = None
        self.samples = None

        if self.ab_test.prior_func == 'beta' and \
                self.ab_test.prior_info == 'specified':
            req_params = ['alpha', 'beta']
            if any(x not in self.ab_test.prior_parameters.keys()
                   for x in req_params):
                raise Exception("""Beta prior parameters must
                                include alpha and beta.""")
            if any(x not in req_params for x in
                   self.ab_test.prior_parameters.keys()):
                print("""WARNING: Beta prior ignoring parameters not
                      alpha or beta.""")
        elif self.ab_test.prior_func == 'poisson' \
                and self.ab_test.prior_info == 'specified':
            req_params = ['alpha', 'beta', 'mean', 'var']
            if ('mean' in self.ab_test.prior_parameters.keys() and
                'var' not in self.ab_test.prior_parameters.keys()) or \
                    ('var' in self.ab_test.prior_parameters.keys() and
                     'mean' not in self.ab_test.prior_parameters.keys()):
                raise Exception("""{} prior parameters must be either mean
                                and var, OR alpha and beta."""
                                .format(self.ab_test.prior_func))
            if ('alpha' in self.ab_test.prior_parameters.keys() and
                'beta' not in self.ab_test.prior_parameters.keys()) or \
                    ('beta' in self.ab_test.prior_parameters.keys() and
                     'alpha' not in self.ab_test.prior_parameters.keys()):
                raise Exception("""{} prior parameters must be either
                                mean and var, OR alpha and beta."""
                                .format(self.ab_test.prior_func))
            if any(x not in req_params for x in
                   self.ab_test.prior_parameters.keys()):
                print('WARNING: {} prior ignoring extra parameters.'
                      .format(self.ab_test.prior_func))
                if 'alpha' in self.ab_test.prior_parameters.keys():
                    print('using parameters alpha: {0} and beta: {1}'
                          .format(self.ab_test.prior_parameters['alpha'],
                                  self.ab_test.prior_parameters['beta']))
                else:
                    print('using parameters mean: {0} and var: {1}'
                          .format(self.ab_test.prior_parameters['mean'],
                                  self.ab_test.prior_parameters['var']))
        elif self.ab_test.prior_info == 'specified':
            req_params = ['mean', 'var']
            if any(x not in self.ab_test.prior_parameters.keys()
                   for x in req_params):
                raise Exception("""{} prior parameters must include mean
                                and var.""".format(self.ab_test.prior_func))
            if any(x not in req_params
                   for x in self.ab_test.prior_parameters.keys()):
                print("""WARNING: {} prior ignoring parameters not mean
                      or var.""".format(self.ab_test.prior_func))

        self.set_params()

    def set_params(self):
        """Process the inputs and set the prior parameters."""
        # pre-processing, required for all
        if self.ab_test.prior_func == 'beta':
            raw_data_agg = self.ab_test.raw_data.groupby(
                                            self.ab_test.bucket_col_name).sum()
            raw_data_agg['bucketed'] = self.ab_test.raw_data.groupby(
                                        self.ab_test
                                        .bucket_col_name).count()[
                                            self.ab_test.metric]
            self.raw_data_agg = raw_data_agg
        elif self.ab_test.prior_func == 'poisson':
            raw_data_agg = self.ab_test.raw_data.groupby(
                           self.ab_test.bucket_col_name).mean()
            raw_data_agg['variance'] = self.ab_test.raw_data.groupby(
                                        self.ab_test.bucket_col_name).var()[
                                           self.ab_test.metric]
            raw_data_agg['bucketed'] = self.ab_test.raw_data.groupby(
                                         self.ab_test.bucket_col_name).count()[
                                             self.ab_test.metric]
            self.raw_data_agg = raw_data_agg
        elif self.ab_test.prior_func in ['log-normal', 'normal']:
            self.control_data = self.ab_test.raw_data \
                                .loc[self.ab_test.raw_data[
                                        self.ab_test.bucket_col_name] ==
                                     self.ab_test.control_bucket_name][
                                        self.ab_test.metric].values
            self.samples = [list(self.control_data)]
            for variant in self.ab_test.variant_bucket_names:
                self.samples.append(list(self.ab_test.raw_data
                                    .loc[self.ab_test.raw_data[
                                         self.ab_test.bucket_col_name] ==
                                         variant][self.ab_test.metric]
                                         .values))

        # user specified
        if self.ab_test.prior_info == 'specified':
            if self.ab_test.prior_func == 'beta':
                self.alpha = self.ab_test.prior_parameters['alpha']
                self.beta = self.ab_test.prior_parameters['beta']
            elif self.ab_test.prior_func == 'poisson':
                if 'mean' in self.ab_test.prior_parameters.keys():
                    mu = self.ab_test.prior_parameters['mean']
                    sigma2 = self.ab_test.prior_parameters['var']
                    sigma2 *= self.ab_test.prior_scale_factor
                    alpha = mu**2/sigma2
                    beta = mu/sigma2
                    self.alpha = alpha
                    self.beta = beta
                else:
                    self.alpha = self.ab_test.prior_parameters['alpha']
                    self.beta = self.ab_test.prior_parameters['beta']
            else:
                self.mean = self.ab_test.prior_parameters['mean']
                self.var = self.ab_test.prior_parameters['var']
            return

        # uninformed
        if self.ab_test.prior_info == 'uninformed':
            if self.ab_test.prior_func == 'beta':
                self.alpha = 1
                self.beta = 1
            elif self.ab_test.prior_info == 'poisson':
                raise Exception('poisson uninformed prior not yet implemented')
            else:
                raise Exception('normal uninformed prior not yet implemented')
            return

        # empirical bayes
        if self.ab_test.prior_func == 'beta':
            prior_converts = self.raw_data_agg.loc[
                                            self.ab_test.control_bucket_name][
                                                self.ab_test.metric]
            prior_non_converts = self.raw_data_agg.loc[
                                            self.ab_test.control_bucket_name][
                                                'bucketed'] - prior_converts
            self.alpha = prior_converts/self.ab_test.prior_scale_factor,
            self.beta = prior_non_converts/self.ab_test.prior_scale_factor,

        elif self.ab_test.prior_func == 'poisson':
            mu = np.mean(self.raw_data_agg[self.ab_test.metric].values)
            sigma2 = np.mean(self.raw_data_agg['variance'].values)
            sigma2 *= self.ab_test.prior_scale_factor
            alpha = mu**2/sigma2
            beta = mu/sigma2
            self.alpha = alpha
            self.beta = beta

        elif self.ab_test.prior_func in ['log-normal', 'normal']:
            if self.ab_test.prior_func == 'log-normal':
                prior_mean = np.mean(np.log(self.control_data))
                prior_var = np.var(np.log(self.control_data))
                prior_var *= self.ab_test.prior_scale_factor
            elif self.ab_test.prior_func == 'normal':
                prior_mean = np.mean(self.control_data)
                prior_var = np.var(self.control_data)
                prior_var *= self.ab_test.prior_scale_factor
            self.mean = prior_mean
            self.var = prior_var
