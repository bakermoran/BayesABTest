"""Distributions functionality for ab_test_model."""

import numpy as np
import pymc3 as pm
#pylint: disable=no-member

class _ab_test_distributions:
  def _get_beta_priors(self, raw_data_agg):
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

  def _get_poisson_priors(self, raw_data_agg):
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


  def _get_normal_priors(self, control_data):
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


  def _set_beta_posteriors(self, prior_distribution, raw_data_agg):
    """Set the pymc3 model to use Binomial posterior sampling."""
    bucketed_accts = raw_data_agg['bucketed'].values
    metric_agg = raw_data_agg[self.metric].values
    pm.Binomial('posterior', n=bucketed_accts, p=prior_distribution,
                observed=metric_agg) # posterior to do: observed may be a bug?


  def _set_poisson_posteriors(self, prior_distribution, raw_data_agg):
    """Set the pymc3 model to use Poisson posterior sampling."""
    metric_agg = raw_data_agg[self.metric].values
    pm.Poisson(name='posterior', mu=prior_distribution, observed=metric_agg)
    # to do: observed may be a bug?


  # http://www.ams.sunysb.edu/~zhu/ams570/Bayesian_Normal.pdf
  # https://www.mhnederlof.nl/bayesnormalupdate.html
  # https://en.wikipedia.org/wiki/Conjugate_prior
  def _norm_update_var(self, prior_var, sample):
    """Update variance to the posterior variance for a (log)
    normal distribution."""
    if self.prior_func == 'log-normal':
      variance = np.var(np.log(sample))
    elif self.prior_func == 'normal':
      variance = np.var(sample)
    posterior_var = ((1/prior_var) + (1/(variance/len(sample))))**(-1)
    return posterior_var


  def _norm_update_mean(self, posterior_var, prior_var,
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


  def _set_normal_posteriors(self, prior_mean, prior_var, samples):
    """Set the pymc3 model to use (log)normal posterior sampling.
    Posteriors are updated as a normal distribution.
    """
    posterior_variances = []
    posterior_means = []
    names = []

    for index, sample in enumerate(samples):
      posterior_var = self._norm_update_var(prior_var, sample)
      posterior_mean = self._norm_update_mean(posterior_var, prior_var,
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


  def _set_distributions(self):
    """Controller that calls correct prior and posterior functions based on
    the distribution called by instantiation.
    """
    if self.prior_func == 'beta':
      raw_data_agg = self.raw_data.groupby(self.bucket_col_name).sum()
      raw_data_agg['bucketed'] = self.raw_data.groupby(
                                  self.bucket_col_name).count()[self.metric]
      prior_distribution = self._get_beta_priors(raw_data_agg)
      self._set_beta_posteriors(prior_distribution, raw_data_agg)
    elif self.prior_func == 'poisson':
      raw_data_agg = self.raw_data.groupby(self.bucket_col_name).mean()
      raw_data_agg['variance'] = self.raw_data.groupby(
                                  self.bucket_col_name).var()[self.metric]
      raw_data_agg['bucketed'] = self.raw_data.groupby(
                                  self.bucket_col_name).count()[self.metric]
      prior_distribution = self._get_poisson_priors(raw_data_agg)
      self._set_poisson_posteriors(prior_distribution, raw_data_agg)
    elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
      C_data = self.raw_data.loc[self.raw_data[self.bucket_col_name] == self.control_bucket_name][self.metric].values
      samples = [list(C_data)]
      for variant in self.variant_bucket_names:
        samples.append(list(self.raw_data.loc[self.raw_data[self.bucket_col_name] == variant][self.metric].values))
      prior_mean, prior_var = self._get_normal_priors(C_data)
      self._set_normal_posteriors(prior_mean, prior_var, samples)


  def _set_samples(self):
    """Set private member variables to the correct sample."""
    if self.prior_func == 'beta' or self.prior_func == 'poisson':
      self.control_sample = self._trace[self.metric][:,0]
      for i in range(1, 1+len(self.variant_bucket_names)):
        self.variant_samples.append(list(self._trace[self.metric][:,i]))
    elif self.prior_func == 'log-normal' or self.prior_func == 'normal':
      for index in range(0, len(self.variant_bucket_names) + 1):
        name = self.metric + str(index)
        if index == 0: self.control_sample = self._trace[name][:,0]
        else: self.variant_samples.append(list(self._trace[name][:,0]))
