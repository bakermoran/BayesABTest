"""Prior distributions for ABTest.

Pass one of these as ``ABTest(prior=...)``:

* ``Uninformed()`` -- a weak default prior for the family.
* ``Empirical(scale=4)`` -- built from the control data, weakened by ``scale``.
* ``Beta`` / ``Gamma`` / ``Normal`` / ``LogNormal`` -- fully specified. Must
  match the family (beta, poisson, normal, log-normal respectively).
"""

from dataclasses import dataclass

import numpy as np


def _positive(**params):
    for name, value in params.items():
        if not value > 0:
            raise ValueError(f'{name} must be greater than 0, got {value}')


@dataclass(frozen=True)
class Beta:
    """Beta(alpha, beta) prior for a conversion rate."""

    alpha: float
    beta: float

    def __post_init__(self):
        """Validate parameters."""
        _positive(alpha=self.alpha, beta=self.beta)


@dataclass(frozen=True)
class Gamma:
    """Gamma(alpha, beta) prior (beta is the rate) for a poisson rate."""

    alpha: float
    beta: float

    def __post_init__(self):
        """Validate parameters."""
        _positive(alpha=self.alpha, beta=self.beta)

    @classmethod
    def from_mean_var(cls, mean, var):
        """Moment match a Gamma to the given mean and variance."""
        _positive(mean=mean, var=var)
        return cls(alpha=mean ** 2 / var, beta=mean / var)


@dataclass(frozen=True)
class Normal:
    """Normal prior on the mean of a normal response."""

    mean: float
    var: float

    def __post_init__(self):
        """Validate parameters."""
        _positive(var=self.var)


@dataclass(frozen=True)
class LogNormal:
    """Normal prior on the mean of the log of a log-normal response.

    ``mean`` and ``var`` are on the log scale.
    """

    mean: float
    var: float

    def __post_init__(self):
        """Validate parameters."""
        _positive(var=self.var)


@dataclass(frozen=True)
class Uninformed:
    """A weak prior with no belief about the response.

    Not available for the poisson family.
    """


@dataclass(frozen=True)
class Empirical:
    """Use the control data as the prior belief, weakened by ``scale``.

    For beta the control counts are divided by ``scale``. For the other
    families the control variance is multiplied by ``scale``.
    """

    scale: float = 4.0

    def __post_init__(self):
        """Validate parameters."""
        _positive(scale=self.scale)


FAMILY_PRIORS = {'beta': Beta, 'poisson': Gamma,
                 'normal': Normal, 'log-normal': LogNormal}


def resolve_prior(prior, family, *, values=None, counts=None):
    """Turn Uninformed / Empirical into a concrete prior for the family.

    Arguments:
        prior -- any of the prior classes in this module.
        family -- 'beta', 'poisson', 'normal' or 'log-normal'.
        values -- control observations (all families but beta).
        counts -- control (successes, trials) (beta only).
    """
    if isinstance(prior, Uninformed):
        if family == 'beta':
            return Beta(1, 1)
        if family == 'normal':
            return Normal(mean=1, var=20000)
        if family == 'log-normal':
            return LogNormal(mean=np.log(1), var=np.log(20000))
        raise ValueError('Uninformed prior is not available for the '
                         f'{family} family; use Empirical() or Gamma()')

    if isinstance(prior, Empirical):
        if family == 'beta':
            successes, trials = counts
            if successes == 0 or successes == trials:
                raise ValueError('Empirical beta prior needs both conversions '
                                 'and non-conversions in the control')
            return Beta(successes / prior.scale,
                        (trials - successes) / prior.scale)
        values = np.asarray(values, dtype=float)
        if family == 'poisson':
            return Gamma.from_mean_var(values.mean(),
                                       values.var(ddof=1) * prior.scale)
        if family == 'normal':
            return Normal(values.mean(), values.var() * prior.scale)
        logs = np.log(values)
        return LogNormal(logs.mean(), logs.var() * prior.scale)

    return prior
