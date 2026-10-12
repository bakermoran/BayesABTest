"""The ABTest class."""

from numbers import Integral

import numpy as np
import pandas as pd

from . import priors as pr
from ._conjugate import beta_posterior, gamma_poisson_posterior, normal_posterior
from .result import Posterior, Result

FAMILIES = ('beta', 'poisson', 'normal', 'log-normal')


class ABTest:
    """Configure a Bayesian AB test, then ``fit`` it to data.

    Example:
        >>> test = ABTest(metric='conversion', family='beta', control='off')
        >>> result = test.fit(df)
        >>> result.prob_better('on')
        >>> result.plot()

    Arguments:
        metric -- column name holding the response variable.
        family -- the distribution of the response:
            'beta' -- conversion rates, binary 0/1 data.
            'poisson' -- discrete counts (arrivals per day, events per
                account, ...).
            'normal' -- continuous on (-inf, +inf).
            'log-normal' -- continuous and greater than zero (premium,
                minutes on scene, ...).

    Keyword Arguments:
        control -- value in the group column for the control bucket
            (default 'off').
        group -- column name holding the bucket (default 'bucket').
        prior -- the prior belief about the response, from
            ``BayesABTest.priors``. Default ``Empirical()``, which uses the
            control as a (weakened) prior. Also ``Uninformed()`` or a fully
            specified ``Beta``/``Gamma``/``Normal``/``LogNormal``.
        n_samples -- number of posterior samples drawn per bucket, at least
            1000 (default 10000).
        random_state -- an int seed or ``numpy.random.Generator``, for
            reproducible samples (default None).
    """

    def __init__(self, metric, family, *, control='off', group='bucket',
                 prior=pr.Empirical(), n_samples=10000, random_state=None):
        """Validate and store the configuration."""
        if family not in FAMILIES:
            raise ValueError(f'family must be one of {list(FAMILIES)}')
        if isinstance(prior, (pr.Uninformed, pr.Empirical)):
            pass
        elif not isinstance(prior, tuple(pr.FAMILY_PRIORS.values())):
            raise TypeError('prior must be a BayesABTest.priors object, '
                            f'got {type(prior).__name__}')
        elif not isinstance(prior, pr.FAMILY_PRIORS[family]):
            raise ValueError(
                f'a {family} family needs a '
                f'{pr.FAMILY_PRIORS[family].__name__} prior, '
                f'got {type(prior).__name__}')
        if not isinstance(n_samples, Integral) or n_samples < 1000:
            raise ValueError('n_samples must be an integer of at least 1000')
        if isinstance(prior, pr.Uninformed) and family == 'poisson':
            raise ValueError('Uninformed prior is not available for the '
                             'poisson family; use Empirical() or Gamma()')

        self.metric = metric
        self.family = family
        self.control = control
        self.group = group
        self.prior = prior
        self.n_samples = n_samples
        self.random_state = random_state

    def __repr__(self):
        """Show the configuration."""
        return (f'ABTest(metric={self.metric!r}, family={self.family!r}, '
                f'control={self.control!r}, group={self.group!r}, '
                f'prior={self.prior!r}, n_samples={self.n_samples})')

    # ------------------------------------------------------------- fitting

    def fit(self, data):
        """Fit the model to a dataframe and return a Result.

        Arguments:
            data -- a pandas DataFrame with (at least) the metric and group
                columns.
        """
        values = self._group_values(data)
        if self.family == 'beta':
            counts = {name: (int(v.sum()), len(v))
                      for name, v in values.items()}
            return self._fit_beta(counts)

        for name, v in values.items():
            if self.family == 'poisson' and (v < 0).any():
                raise ValueError('poisson data must be non-negative')
            if self.family == 'log-normal' and (v <= 0).any():
                raise ValueError('log-normal data must be greater than 0')
            if self.family in ('normal', 'log-normal') and len(v) < 2:
                raise ValueError(f'bucket {name!r} needs at least 2 rows')
        prior = pr.resolve_prior(self.prior, self.family,
                                 values=values[self.control])
        params = {}
        for name, v in values.items():
            if self.family == 'poisson':
                alpha, beta = gamma_poisson_posterior(
                    prior.alpha, prior.beta, v.sum(), len(v))
                params[name] = {'alpha': alpha, 'beta': beta}
            else:
                data_ = np.log(v) if self.family == 'log-normal' else v
                mean, var = normal_posterior(prior.mean, prior.var, data_)
                params[name] = {'mean': mean, 'var': var}
        return self._result(prior, params)

    def fit_counts(self, successes, trials):
        """Fit a beta model directly from aggregated counts.

        Arguments:
            successes -- dict of bucket name to number of conversions.
            trials -- dict of bucket name to number of units bucketed.
        """
        if self.family != 'beta':
            raise ValueError('fit_counts is only available for the beta '
                             'family')
        if set(successes) != set(trials):
            raise ValueError('successes and trials must have the same '
                             'bucket names')
        if self.control not in trials:
            raise ValueError(f'control {self.control!r} not found in counts')
        if len(trials) < 2:
            raise ValueError('at least one variant besides the control is '
                             'required')
        for name in trials:
            if not 0 <= successes[name] <= trials[name] or trials[name] < 1:
                raise ValueError(f'bucket {name!r} needs 0 <= successes <= '
                                 'trials and trials >= 1')
        order = [self.control] + [n for n in trials if n != self.control]
        return self._fit_beta({n: (successes[n], trials[n]) for n in order})

    def _group_values(self, data):
        """Validate data and return bucket name -> metric values, control first."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError('data must be a pandas DataFrame')
        if data.empty:
            raise ValueError('Input dataframe must contain data')
        for col in (self.metric, self.group):
            if col not in data.columns:
                raise ValueError(f'Input dataframe must contain column: {col}')
        if data[[self.metric, self.group]].isna().any().any():
            raise ValueError('metric and group columns must not contain '
                             'missing values')
        if self.control not in set(data[self.group]):
            raise ValueError(f'Input dataframe {self.group} column must '
                             f'contain {self.control!r} as the control bucket')
        values = {name: g[self.metric].to_numpy()
                  for name, g in data.groupby(self.group, sort=False)}
        if len(values) < 2:
            raise ValueError('at least one variant besides the control is '
                             'required')
        if self.family == 'beta' and not all(
                np.isin(v, [0, 1]).all() for v in values.values()):
            raise ValueError('Observed data for beta family should be binary')
        values = {name: np.asarray(v, dtype=float)
                  for name, v in values.items()}
        return {self.control: values.pop(self.control), **values}

    def _fit_beta(self, counts):
        prior = pr.resolve_prior(self.prior, 'beta',
                                 counts=counts[self.control])
        params = {}
        for name, (successes, trials) in counts.items():
            alpha, beta = beta_posterior(prior.alpha, prior.beta,
                                         successes, trials)
            params[name] = {'alpha': alpha, 'beta': beta}
        return self._result(prior, params)

    def _result(self, prior, params):
        rng = np.random.default_rng(self.random_state)
        samples = {name: self._draw(rng, p) for name, p in params.items()}
        posteriors = {name: Posterior(name, self.family, p)
                      for name, p in params.items()}
        return Result(metric=self.metric, family=self.family,
                      control=self.control,
                      variants=[n for n in params if n != self.control],
                      prior=prior, posteriors=posteriors,
                      samples=pd.DataFrame(samples))

    def _draw(self, rng, p):
        size = self.n_samples
        if self.family == 'beta':
            return rng.beta(p['alpha'], p['beta'], size)
        if self.family == 'poisson':
            return rng.gamma(p['alpha'], 1 / p['beta'], size)
        draws = rng.normal(p['mean'], np.sqrt(p['var']), size)
        return np.exp(draws) if self.family == 'log-normal' else draws
