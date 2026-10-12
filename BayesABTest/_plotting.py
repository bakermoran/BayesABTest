"""Plotting for Result. Public entry points live on Result."""

import textwrap
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator, PercentFormatter

_VARIANT_COLORS = ['r', 'g', 'y', 'm', 'c', 'darkred', 'darkorange',
                   'indigo', 'slategrey', 'coral']


def _colors(result):
    colors = {result.control: 'blue'}
    for i, name in enumerate(result.variants):
        colors[name] = _VARIANT_COLORS[i % len(_VARIANT_COLORS)]
    return colors


def _join_names(names):
    names = [str(n) for n in names]
    if len(names) <= 2:
        return ' and '.join(names)
    return ', '.join(names[:-1]) + ', and ' + names[-1]


def _title(string):
    return '\n'.join(textwrap.wrap(string, 40))


def _percent_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=None))


def plot_posteriors(result, variants=None, ax=None):
    """Plot a KDE of the posterior samples."""
    ax = ax if ax is not None else plt.gca()
    variants = list(result.samples.columns) if variants is None \
        else list(variants)
    colors = _colors(result)
    for name in variants:
        if name not in colors:
            raise ValueError(f'{name!r} is not a bucket; expected one of '
                             f'{list(colors)}')
        sns.kdeplot(result._draws(name), fill=True, color=colors[name], ax=ax)
    ax.legend(labels=[str(v) for v in variants], loc='upper right')
    ax.set_xlabel('Conversion Rate' if result.family == 'beta'
                  else result.metric)
    sns.despine(ax=ax, left=True)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_title(_title('Distribution(s) for {0} for {1}'.format(
        _join_names(variants), result.metric)))
    if result.family == 'beta':
        _percent_axis(ax)
    return ax


def plot_lift(result, numerator, denominator=None, ax=None):
    """Plot the lift as a KDE, titled with the probability of positive lift."""
    ax = ax if ax is not None else plt.gca()
    denominator = result.control if denominator is None else denominator
    lift = result.lift(numerator, denominator)
    sns.kdeplot(lift, ax=ax)
    # shade the density, highlighting the proportion with positive lift
    x, y = ax.lines[-1].get_data()
    ax.fill_between(x, y, where=x <= 0, color='C0', alpha=.25,
                    interpolate=True, linewidth=0)
    ax.fill_between(x, y, where=x > 0, color='red', alpha=.5,
                    interpolate=True, linewidth=0)
    if len(result.variants) > 1:
        ax.set_ylabel(f'{numerator} vs {denominator}', rotation=0,
                      fontstyle='italic')
    else:
        ax.set_ylabel('')
    ax.axvline(x=0, linestyle='dotted', color='black')
    ax.set_xlabel('Lift')
    ax.set_title(_title('{0} had {1:.2%} probability of positive lift'.format(
        result.metric, result.prob_better(numerator, denominator))))
    sns.despine(ax=ax, left=True)
    ax.set_yticks([])
    _percent_axis(ax)
    return ax


def plot_ecdf(result, numerator, denominator=None, ci=0.95, ax=None):
    """Plot the lift ECDF with the median and credible interval."""
    ax = ax if ax is not None else plt.gca()
    denominator = result.control if denominator is None else denominator
    x, y = result.ecdf(numerator, denominator)
    low, high = result.credible_interval(numerator, denominator, ci)
    median = float(np.median(result.lift(numerator, denominator)))
    ax.plot(x, y)
    ax.set_title(_title('Median Lift was {0:.2%}, with a {1:.0%} CI of '
                        '[{2:.2%}, {3:.2%}]'.format(median, ci, low, high)))
    ax.set_xlabel('Lift')
    ax.set_ylabel('Cumulative Probability')
    for value in (low, median, high):
        ax.axvline(x=value, linestyle='dotted', color='black')
    sns.despine(ax=ax, left=True)
    _percent_axis(ax)
    return ax


def plot_report(result, lift, compare_variants, ci):
    """Create the multi-chart report figure."""
    variants = result.variants
    if compare_variants and len(variants) < 2:
        raise ValueError('compare_variants requires at least 2 variants')
    if lift and compare_variants and len(variants) > 3:
        lift = False
        warnings.warn('lift plots are turned off due to the large number '
                      'of variant comparisons. Use fewer variants or '
                      'compare_variants=False.', stacklevel=3)
    if lift and len(variants) > 5:
        lift = False
        warnings.warn('lift plots are turned off for more than 5 variants. '
                      'Only the posterior PDFs will be plotted.',
                      stacklevel=3)

    fig = plt.figure(figsize=[12.8, 9.6])
    fig.suptitle(f'Bayesian AB Test Report for {result.metric}', fontsize=16,
                 fontweight='bold')
    fig.subplots_adjust(hspace=1)

    if not lift:
        plot_posteriors(result, ax=fig.add_subplot(1, 1, 1))
    elif len(variants) == 1:
        plot_posteriors(result, ax=fig.add_subplot(2, 2, 1))
        plot_lift(result, variants[0], ax=fig.add_subplot(2, 2, 2))
        plot_ecdf(result, variants[0], ci=ci, ax=fig.add_subplot(2, 1, 2))
    else:
        pairs = [(v, result.control) for v in variants]
        if compare_variants:
            pairs += [(variants[j], variants[i])
                      for i, j in combinations(range(len(variants)), 2)]
        nrows = len(pairs) + 1
        plot_posteriors(result, ax=fig.add_subplot(nrows, 1, 1))
        for row, (num, den) in enumerate(pairs):
            loc = 3 + 2 * row
            plot_lift(result, num, den, ax=fig.add_subplot(nrows, 2, loc))
            plot_ecdf(result, num, den, ci=ci,
                      ax=fig.add_subplot(nrows, 2, loc + 1))
    return fig
