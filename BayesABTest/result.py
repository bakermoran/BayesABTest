"""Result objects returned by ABTest.fit()."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import _plotting
from ._conjugate import expected_loss_beta


@dataclass(frozen=True)
class Posterior:
    """The fitted posterior distribution for one bucket.

    ``params`` holds ``alpha``/``beta`` (beta and poisson families) or
    ``mean``/``var`` of the mean (normal and log-normal; for log-normal these
    are on the log scale).
    """

    name: object
    family: str
    params: dict


@dataclass(frozen=True)
class LossResult:
    """Outcome of an expected loss comparison of a variant to the control."""

    control: object
    variant: object
    loss_type: str
    epsilon: float
    risk_control: float
    risk_variant: float
    stop: bool
    winner: object

    def __str__(self):
        """Human readable summary."""
        if self.stop:
            head = (f'Test can be stopped and {self.winner} can be declared '
                    'the winner')
        else:
            head = 'Keep running the test'
        return (f'{head}\n'
                f'risk of choosing {self.control} = {self.risk_control}\n'
                f'risk of choosing {self.variant} = {self.risk_variant}')


@dataclass(frozen=True, repr=False)
class Result:
    """The fitted model: posterior samples plus lift, loss and plotting.

    Attributes:
        metric -- the response column name.
        family -- 'beta', 'poisson', 'normal' or 'log-normal'.
        control -- the control bucket name.
        variants -- the other bucket names.
        prior -- the concrete prior used (after resolving Empirical etc).
        posteriors -- dict of bucket name to Posterior.
        samples -- DataFrame of posterior samples, one column per bucket.
    """

    metric: str
    family: str
    control: object
    variants: list
    prior: object
    posteriors: dict
    samples: pd.DataFrame = field(compare=False)

    def __repr__(self):
        """Short representation, without dumping the samples."""
        return (f'Result(metric={self.metric!r}, family={self.family!r}, '
                f'control={self.control!r}, variants={self.variants!r}, '
                f'n_samples={len(self.samples)})')

    # ------------------------------------------------------------- numbers

    def _draws(self, name):
        if name not in self.samples.columns:
            raise ValueError(f'{name!r} is not a bucket; expected one of '
                             f'{list(self.samples.columns)}')
        return self.samples[name].to_numpy()

    def lift(self, numerator, denominator=None):
        """Lift of numerator over denominator for each sample.

        ``numerator / denominator - 1``. The denominator defaults to the
        control.
        """
        denominator = self.control if denominator is None else denominator
        if numerator == denominator:
            raise ValueError('numerator and denominator cannot be the same')
        return self._draws(numerator) / self._draws(denominator) - 1

    def prob_better(self, numerator, denominator=None):
        """Probability the lift of numerator over denominator is positive."""
        return float(np.mean(self.lift(numerator, denominator) > 0))

    def credible_interval(self, numerator, denominator=None, level=0.95):
        """Central credible interval (low, high) for the lift."""
        if not 0 < level < 1:
            raise ValueError('level must be in the interval (0, 1)')
        tail = (1 - level) / 2
        low, high = np.quantile(self.lift(numerator, denominator),
                                [tail, 1 - tail])
        return float(low), float(high)

    def ecdf(self, numerator, denominator=None):
        """Empirical CDF of the lift. Returns arrays ``(x, y)``."""
        lift = np.sort(self.lift(numerator, denominator))
        x = np.unique(lift)
        y = np.searchsorted(lift, x, side='right') / lift.size
        return x, y

    def summary(self, level=0.95):
        """DataFrame summarizing each bucket.

        Columns are the posterior mean and credible interval of the metric
        and, for each variant, the median lift over control and the
        probability of positive lift. Control rows have NaN lift columns.
        """
        tail = (1 - level) / 2
        rows = {}
        for name in self.samples.columns:
            draws = self._draws(name)
            low, high = np.quantile(draws, [tail, 1 - tail])
            row = {'mean': draws.mean(), 'ci_low': low, 'ci_high': high,
                   'median_lift': np.nan, 'prob_better': np.nan}
            if name != self.control:
                row['median_lift'] = np.median(self.lift(name))
                row['prob_better'] = self.prob_better(name)
            rows[name] = row
        return pd.DataFrame.from_dict(rows, orient='index')

    def loss(self, variant, *, loss_type='absolute', epsilon=0.0001):
        """Compute the expected loss of choosing each of control and variant.

        Only available for the beta family. The test can be stopped once the
        risk of choosing one side falls to ``epsilon`` or below.

        Arguments:
            variant -- the variant to compare to the control.
            loss_type -- 'absolute' or 'percent' (loss in lift terms).
            epsilon -- the risk threshold for stopping.
        """
        if self.family != 'beta':
            raise ValueError('loss is only supported for the beta family')
        if variant == self.control or variant not in self.variants:
            raise ValueError(f'{variant!r} must be one of the variants: '
                             f'{self.variants}')
        ctrl = self.posteriors[self.control].params
        var = self.posteriors[variant].params
        a, b = ctrl['alpha'], ctrl['beta']
        c, d = var['alpha'], var['beta']
        # risk of choosing the control is the variant's gain over it
        risk_control = expected_loss_beta(c, d, a, b, loss_type)
        risk_variant = expected_loss_beta(a, b, c, d, loss_type)
        if risk_control <= epsilon:
            stop, winner = True, self.control
        elif risk_variant <= epsilon:
            stop, winner = True, variant
        else:
            stop, winner = False, None
        return LossResult(self.control, variant, loss_type, epsilon,
                          risk_control, risk_variant, stop, winner)

    # ------------------------------------------------------------ plotting

    def plot_posteriors(self, variants=None, *, ax=None):
        """Plot the posterior densities. Returns the Axes.

        Arguments:
            variants -- bucket names to plot (default all).
            ax -- matplotlib Axes to draw on (default current Axes).
        """
        return _plotting.plot_posteriors(self, variants, ax)

    def plot_lift(self, numerator, denominator=None, *, ax=None):
        """Plot the lift density and probability of positive lift."""
        return _plotting.plot_lift(self, numerator, denominator, ax)

    def plot_ecdf(self, numerator, denominator=None, *, ci=0.95, ax=None):
        """Plot the lift ECDF with the median and ``ci`` credible interval."""
        return _plotting.plot_ecdf(self, numerator, denominator, ci, ax)

    def plot(self, *, lift=True, compare_variants=False, ci=0.95):
        """Create the full report figure and return it.

        Arguments:
            lift -- also plot the lift density and ECDF (default True).
            compare_variants -- also compare variants to each other, not just
                to the control (default False).
            ci -- credible interval level shown on the ECDF (default 0.95).
        """
        return _plotting.plot_report(self, lift, compare_variants, ci)
