"""Sampling functionality for ab_test_model."""

import numpy as np
from scipy import stats
# pylint: disable=no-member


class _ab_test_model_sampling:
    """Mixin class to provide sampling functionality to the model."""

    def _sample_beta(self):
        """Sample a beta distribution and set the trace variables."""
        self._trace[self.metric] = {}

        for name, posterior in self.posteriors.items():
            posterior_alpha = self.prior_params.alpha + \
                posterior.get_metric_agg()
            posterior_beta = self.prior_params.beta + \
                posterior.get_bucketed() - posterior.get_metric_agg()
            self._trace[self.metric][name] = stats.beta.rvs(posterior_alpha,
                                                            posterior_beta,
                                                            size=self.samples)

    def _sample_poisson(self):
        """Sample a poisson distribution and set the trace variables."""
        self._trace[self.metric] = {}

        for name, posterior in self.posteriors.items():
            self._trace[self.metric][name] = stats.gamma.rvs(
                posterior.get_metric_agg(), size=self.samples)

    def _norm_update_var(self, prior_var, sample):
        """Update var to the post var for a (log) normal distribution."""
        if self.prior_function == 'log-normal':
            variance = np.var(np.log(sample))
        elif self.prior_function == 'normal':
            variance = np.var(sample)
        posterior_var = ((1/prior_var) + (1/(variance/len(sample))))**(-1)
        return posterior_var

    def _norm_update_mean(self, posterior_var, prior_var,
                          prior_mean, sample):
        """Update mean to the post mean for a (log) normal distribution."""
        if self.prior_function == 'log-normal':
            mean = np.mean(np.log(sample))
            variance = np.var(np.log(sample))
        elif self.prior_function == 'normal':
            mean = np.mean(sample)
            variance = np.var(sample)
        posterior_mean = posterior_var*((prior_mean/prior_var) +
                                        (mean/(variance/len(sample))))
        return posterior_mean

    def _sample_normal(self):
        """Sample a (log) normal distribution and set the trace variables."""
        self._trace[self.metric] = {}

        for name, posterior in self.posteriors.items():
            posterior_var = self._norm_update_var(
                self.prior_params.var, posterior.get_likelihood_sample())
            posterior_mean = self._norm_update_mean(
                posterior_var, self.prior_params.var, self.prior_params.mean,
                posterior.get_likelihood_sample())
            posterior.set_posterior_mean(posterior_mean)
            posterior.set_posterior_var(posterior_var)
            posterior.set_posterior_sample_name(self.metric + '_' + name)
            if self.prior_function == 'log-normal':
                self._trace[self.metric][name] = np.random.lognormal(
                    posterior.get_posterior_mean(),
                    np.sqrt(posterior.get_posterior_var()), self.samples)
            elif self.prior_function == 'normal':
                self._trace[self.metric][name] = stats.norm.rvs(
                    loc=posterior.get_posterior_mean(),
                    scale=np.sqrt(posterior.get_posterior_var()),
                    size=self.samples)

    def _sample(self):
        """Control that calls correct prior and posterior functions.

        Based on the distribution called by instantiation.
        """
        if self.prior_function == 'beta':
            self._sample_beta()

        elif self.prior_function == 'poisson':
            self._sample_poisson()

        elif (self.prior_function == 'log-normal'
              or self.prior_function == 'normal'):
            self._sample_normal()

        for name, posterior in self.posteriors.items():
            posterior.set_posterior_sample(
                                    list(self._trace[self.metric][name]))
