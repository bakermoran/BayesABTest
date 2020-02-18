"""Distributions functionality for ab_test_model."""

import numpy as np
import pymc3 as pm
# pylint: disable=no-member


class _ab_test_distributions:
    """Mixin class to provide distribution functionality to the model."""

    def _get_beta_priors(self):
        """Return prior distribution of a beta distribution.

        This returns either informed or uninformed depending on input.
        """
        num_vars = 1 + len(self.variant_bucket_names)
        prior_distribution = pm.Beta(self.metric,
                                     alpha=self.prior_params.alpha,
                                     beta=self.prior_params.beta,
                                     shape=num_vars)
        return prior_distribution

    def _get_poisson_priors(self):
        """Return prior distribution of a gamma distribution.

        This returns either informed or uninformed depending on input.
        """
        num_vars = 1 + len(self.variant_bucket_names)
        prior_distribution = pm.Gamma(self.metric,
                                      alpha=self.prior_params.alpha,
                                      beta=self.prior_params.beta,
                                      shape=num_vars)
        return prior_distribution

    def _get_normal_priors(self):
        """Return prior distribution parameters of a (log)normal distribution.

        This returns either informed or uninformed depending on input.
        """
        # This is preset by the prior_params
        return 0

    def _set_beta_posteriors(self, prior_distribution):
        """Set the pymc3 model to use Binomial posterior sampling."""
        pm.Binomial('posterior',
                    n=self.prior_params.raw_data_agg['bucketed'].values,
                    p=prior_distribution,
                    observed=self.prior_params.raw_data_agg[
                                                           self.metric].values)

    def _set_poisson_posteriors(self, prior_distribution):
        """Set the pymc3 model to use Poisson posterior sampling."""
        pm.Poisson(name='posterior', mu=prior_distribution,
                   observed=self.prior_params.raw_data_agg[self.metric].values)

    def _norm_update_var(self, prior_var, sample):
        """Update var to the post var for a (log) normal distribution."""
        if self.prior_func == 'log-normal':
            variance = np.var(np.log(sample))
        elif self.prior_func == 'normal':
            variance = np.var(sample)
        posterior_var = ((1/prior_var) + (1/(variance/len(sample))))**(-1)
        return posterior_var

    def _norm_update_mean(self, posterior_var, prior_var,
                          prior_mean, sample):
        """Update mean to the post mean for a (log) normal distribution."""
        if self.prior_func == 'log-normal':
            mean = np.mean(np.log(sample))
            variance = np.var(np.log(sample))
        elif self.prior_func == 'normal':
            mean = np.mean(sample)
            variance = np.var(sample)
        posterior_mean = posterior_var*((prior_mean/prior_var) +
                                        (mean/(variance/len(sample))))
        return posterior_mean

    def _set_normal_posteriors(self, prior_distribution):
        """Set the pymc3 model to use (log) normal posterior sampling."""
        posterior_variances = []
        posterior_means = []
        names = []

        for index, sample in enumerate(self.prior_params.samples):
            posterior_var = self._norm_update_var(self.prior_params.var,
                                                  sample)
            posterior_mean = self._norm_update_mean(posterior_var,
                                                    self.prior_params.var,
                                                    self.prior_params.mean,
                                                    sample)
            posterior_variances.append(posterior_var)
            posterior_means.append(posterior_mean)
            names.append(self.metric + str(index))

        for var, mean, name in zip(posterior_variances,
                                   posterior_means,
                                   names):
            if self.prior_func == 'log-normal':
                pm.Lognormal(name=name,
                             mu=mean,
                             sigma=np.sqrt(var),
                             shape=1)
            elif self.prior_func == 'normal':
                pm.Normal(name=name,
                          mu=mean,
                          sigma=np.sqrt(var),
                          shape=1)

    def _set_distributions(self):
        """Control that calls correct prior and posterior functions.

        Based on the distribution called by instantiation.
        """
        if self.prior_func == 'beta':
            prior_distribution = self._get_beta_priors()
            self._set_beta_posteriors(prior_distribution)

        elif self.prior_func == 'poisson':
            prior_distribution = self._get_poisson_priors()
            self._set_poisson_posteriors(prior_distribution)

        elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
            prior_distribution = self._get_normal_priors()
            self._set_normal_posteriors(prior_distribution)

    def _set_samples(self):
        """Set private member variables to the correct sample."""
        if self.prior_func == 'beta' or self.prior_func == 'poisson':
            self.control_sample = self._trace[self.metric][:, 0]
            for i in range(1, 1 + len(self.variant_bucket_names)):
                self.variant_samples.append(list(
                                            self._trace[self.metric][:, i]))
        elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
            for index in range(0, len(self.variant_bucket_names) + 1):
                name = self.metric + str(index)
                if index == 0:
                    self.control_sample = self._trace[name][:, 0]
                else:
                    self.variant_samples.append(list(self._trace[name][:, 0]))
