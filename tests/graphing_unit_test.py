"""Unit tests for the Result plotting methods."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PAIRS = [('variant_1', 'control'), ('variant_2', 'variant_1'),
         ('variant_2', 'control'), ('control', 'variant_1')]


@pytest.mark.parametrize('variants', [
    None,
    ['control', 'variant_1', 'variant_2'],
    ['variant_1', 'variant_2'],
    ['control', 'variant_2'],
    ['variant_1'],
    ['control'],
])
def test_plot_posteriors_subsets(fitted_poisson_result, variants):
    ax = fitted_poisson_result.plot_posteriors(variants)
    assert isinstance(ax, Axes)
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == (variants or ['control', 'variant_1', 'variant_2'])


def test_plot_posteriors_rejects_unknown_variant(fitted_poisson_result):
    with pytest.raises(ValueError):
        fitted_poisson_result.plot_posteriors(['nope'])


@pytest.mark.parametrize('num, den', PAIRS)
def test_plot_lift_reports_probability_of_positive_lift(
        fitted_poisson_result, num, den):
    ax = fitted_poisson_result.plot_lift(num, den)
    expected = fitted_poisson_result.prob_better(num, den)
    assert f'{expected:.2%}' in ax.get_title()


@pytest.mark.parametrize('num, den', PAIRS)
def test_plot_ecdf_title_matches_data(fitted_poisson_result, num, den):
    ax = fitted_poisson_result.plot_ecdf(num, den, ci=0.9)
    lift = fitted_poisson_result.lift(num, den)
    low, high = fitted_poisson_result.credible_interval(num, den, 0.9)
    assert '90% CI' in ax.get_title()
    assert f'[{low:.2%}, {high:.2%}]' in ax.get_title()
    x, y = ax.lines[0].get_data()
    assert y[-1] == 1.0 and x[0] == lift.min()


def test_plots_draw_on_a_provided_axes(fitted_poisson_result):
    fig, (left, right) = plt.subplots(1, 2)
    assert fitted_poisson_result.plot_lift('variant_1', ax=left) is left
    assert fitted_poisson_result.plot_ecdf('variant_1', ax=right) is right


def test_default_denominator_is_control(fitted_poisson_result):
    explicit = fitted_poisson_result.plot_lift('variant_1', 'control')
    default = fitted_poisson_result.plot_lift('variant_1')
    assert explicit.get_title() == default.get_title()


@pytest.mark.parametrize('kwargs, n_axes', [
    ({'lift': False}, 1),
    ({}, 1 + 2 * 2),                              # posteriors + 2 variants
    ({'compare_variants': True}, 1 + 2 * 3),      # + variant_1 vs variant_2
])
def test_report_layout(fitted_poisson_result, kwargs, n_axes):
    fig = fitted_poisson_result.plot(**kwargs)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == n_axes


def test_report_compare_variants_needs_two_variants():
    import numpy as np
    import pandas as pd

    from BayesABTest import ABTest
    df = pd.DataFrame({'bucket': ['off', 'on'] * 100,
                       'conv': np.random.default_rng(0).binomial(1, .5, 200)})
    result = ABTest('conv', 'beta').fit(df)
    assert len(result.plot().axes) == 3           # the one variant layout
    with pytest.raises(ValueError):
        result.plot(compare_variants=True)


def test_report_turns_lift_off_for_many_variants():
    import numpy as np
    import pandas as pd

    from BayesABTest import ABTest
    rng = np.random.default_rng(0)
    names = [f'v{i}' for i in range(7)]
    df = pd.DataFrame({'bucket': np.repeat(names, 50),
                       'conv': rng.binomial(1, .5, 350)})
    result = ABTest('conv', 'beta', control='v0').fit(df)
    with pytest.warns(UserWarning, match='more than 5 variants'):
        fig = result.plot()
    assert len(fig.axes) == 1


def test_plot_lift_highlights_only_the_positive_region(fitted_poisson_result):
    ax = fitted_poisson_result.plot_lift('variant_2')
    highlight = ax.collections[-1]
    assert highlight.get_facecolor()[0][0] == 1.0       # red
    assert highlight.get_paths()[0].vertices[:, 0].min() >= -1e-9
