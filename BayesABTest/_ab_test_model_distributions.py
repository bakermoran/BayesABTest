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
        size = []
        obs = []
        for posterior in self.posteriors.values():
            size.append(posterior.get_bucketed())
            obs.append(posterior.get_metric_agg())
        pm.Binomial('posterior',
                    n=size,
                    p=prior_distribution,
                    observed=obs)

    def _set_poisson_posteriors(self, prior_distribution):
        """Set the pymc3 model to use Poisson posterior sampling."""
        obs = []
        for posterior in self.posteriors.values():
            obs.append(posterior.get_metric_agg())
        pm.Poisson(name='posterior', mu=prior_distribution,
                   observed=obs)

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
        for key, val in self.posteriors.items():
            posterior_var = self._norm_update_var(self.prior_params.var,
                                                  val.get_likelihood_sample())
            posterior_mean = self._norm_update_mean(
                                                 posterior_var,
                                                 self.prior_params.var,
                                                 self.prior_params.mean,
                                                 val.get_likelihood_sample())
            val.set_posterior_mean(posterior_mean)
            val.set_posterior_var(posterior_var)
            val.set_posterior_sample_name(self.metric + '_' + key)
            if self.prior_func == 'log-normal':
                pm.Lognormal(name=val.get_posterior_sample_name(),
                             mu=val.get_posterior_mean(),
                             sd=np.sqrt(val.get_posterior_var()),
                             shape=1)
            elif self.prior_func == 'normal':
                pm.Normal(name=val.get_posterior_sample_name(),
                          mu=val.get_posterior_mean(),
                          sd=np.sqrt(val.get_posterior_var()),
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
            for key, val in self.posteriors.items():
                index = list(self.posteriors.keys()).index(key)
                val.set_posterior_sample(
                                      list(self._trace[self.metric][:, index]))
        elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
            for key, val in self.posteriors.items():
                val.set_posterior_sample(list(self._trace[
                                       val.get_posterior_sample_name()][:, 0]))
