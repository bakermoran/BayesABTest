"""Tests for the ABTest public API: validation, reproducibility, counts."""
import numpy as np
import pandas as pd
import pytest

from BayesABTest import ABTest, Result, priors


def _df(n=300, seed=0, buckets=('off', 'on')):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'bucket': np.repeat(list(buckets), n),
        'conv': rng.binomial(1, .1, n * len(buckets)),
        'count': rng.poisson(3, n * len(buckets)),
        'value': rng.lognormal(1, .5, n * len(buckets))})


# ------------------------------------------------------------ config

@pytest.mark.parametrize('kwargs', [
    {'family': 'gamma'},
    {'n_samples': 999},
    {'n_samples': 1000.5},
    {'prior': priors.Normal(0, 1)},                # wrong family
    {'family': 'poisson', 'prior': priors.Uninformed()},
])
def test_invalid_config_raises_value_error(kwargs):
    args = dict(metric='conv', family='beta')
    args.update(kwargs)
    with pytest.raises(ValueError):
        ABTest(**args)


def test_non_prior_object_raises_type_error():
    with pytest.raises(TypeError):
        ABTest('conv', 'beta', prior={'alpha': 1, 'beta': 1})


@pytest.mark.parametrize('factory', [
    lambda: priors.Beta(0, 1), lambda: priors.Gamma(1, -1),
    lambda: priors.Normal(0, 0), lambda: priors.LogNormal(0, -2),
    lambda: priors.Empirical(0), lambda: priors.Gamma.from_mean_var(3, 0)])
def test_invalid_prior_parameters_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_config_is_inspectable():
    test = ABTest('conv', 'beta', control='ctrl')
    assert "metric='conv'" in repr(test) and "control='ctrl'" in repr(test)
    assert test.prior == priors.Empirical()


# -------------------------------------------------------------- data

def test_fit_returns_result_with_one_column_per_bucket():
    result = ABTest('conv', 'beta', n_samples=1500).fit(_df())
    assert isinstance(result, Result)
    assert list(result.samples.columns) == ['off', 'on']
    assert result.samples.shape == (1500, 2)
    assert result.control == 'off' and result.variants == ['on']
    assert 'n_samples=1500' in repr(result)


def test_control_column_is_first_regardless_of_data_order():
    df = _df(buckets=('on', 'other', 'off'))
    result = ABTest('conv', 'beta').fit(df)
    assert list(result.samples.columns) == ['off', 'on', 'other']
    assert result.variants == ['on', 'other']


def test_custom_column_and_control_names():
    df = _df().rename(columns={'bucket': 'arm'}).replace({'off': 'a'})
    result = ABTest('conv', 'beta', control='a', group='arm').fit(df)
    assert result.control == 'a'


def test_fit_does_not_mutate_input():
    df = _df()
    before = df.copy()
    ABTest('conv', 'beta').fit(df)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize('mutate, family, metric', [
    (lambda d: d.iloc[0:0], 'beta', 'conv'),                  # empty
    (lambda d: d.drop(columns='conv'), 'beta', 'conv'),       # no metric
    (lambda d: d.drop(columns='bucket'), 'beta', 'conv'),     # no group
    (lambda d: d[d.bucket == 'on'], 'beta', 'conv'),          # no control
    (lambda d: d[d.bucket == 'off'], 'beta', 'conv'),         # no variant
    (lambda d: d.assign(conv=d.conv.replace({1: 2})), 'beta', 'conv'),
    (lambda d: d.assign(conv=d.conv.astype(float).where(
        d.index != 0)), 'beta', 'conv'),                      # NaN
    (lambda d: d.assign(count=-d['count']), 'poisson', 'count'),
    (lambda d: d.assign(value=-d.value), 'log-normal', 'value'),
])
def test_bad_data_raises_value_error(mutate, family, metric):
    with pytest.raises(ValueError):
        ABTest(metric, family).fit(mutate(_df()))


def test_fit_requires_a_dataframe():
    with pytest.raises(TypeError):
        ABTest('conv', 'beta').fit([1, 0, 1])


def test_more_than_ten_variants_is_allowed():
    names = [f'v{i}' for i in range(12)]
    result = ABTest('conv', 'beta', control='v0').fit(
        _df(buckets=names, n=50))
    assert len(result.variants) == 11
    result.plot(lift=False)


# ----------------------------------------------------- reproducibility

def test_random_state_makes_samples_reproducible():
    df = _df()
    first = ABTest('conv', 'beta', random_state=42).fit(df).samples
    second = ABTest('conv', 'beta', random_state=42).fit(df).samples
    other = ABTest('conv', 'beta', random_state=43).fit(df).samples
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_random_state_accepts_a_generator():
    rng = np.random.default_rng(1)
    result = ABTest('conv', 'beta', random_state=rng).fit(_df())
    assert result.samples.notna().all().all()


def test_library_does_not_print(capsys):
    result = ABTest('conv', 'beta', random_state=0).fit(_df())
    result.lift('on')
    result.loss('on')
    result.summary()
    assert capsys.readouterr().out == ''


# ------------------------------------------------------------ counts

def test_fit_counts_matches_fit_on_raw_data():
    df = _df()
    counts = df.groupby('bucket').conv.agg(['sum', 'count'])
    a = ABTest('conv', 'beta', random_state=0).fit(df)
    b = ABTest('conv', 'beta', random_state=0).fit_counts(
        successes=counts['sum'].to_dict(), trials=counts['count'].to_dict())
    assert a.posteriors == b.posteriors
    pd.testing.assert_frame_equal(a.samples, b.samples)


@pytest.mark.parametrize('successes, trials', [
    ({'off': 1, 'on': 2}, {'off': 10}),                 # mismatched keys
    ({'off': 11, 'on': 2}, {'off': 10, 'on': 10}),      # more than trials
    ({'off': -1, 'on': 2}, {'off': 10, 'on': 10}),
    ({'off': 1}, {'off': 10}),                          # no variant
    ({'on': 1, 'x': 2}, {'on': 10, 'x': 10}),           # no control
])
def test_fit_counts_validates(successes, trials):
    with pytest.raises(ValueError):
        ABTest('conv', 'beta').fit_counts(successes, trials)


def test_fit_counts_is_beta_only():
    with pytest.raises(ValueError):
        ABTest('count', 'poisson').fit_counts({'off': 1, 'on': 2},
                                              {'off': 5, 'on': 5})


# ------------------------------------------------------------ result

def test_result_methods_validate_bucket_names():
    result = ABTest('conv', 'beta').fit(_df())
    with pytest.raises(ValueError):
        result.lift('nope')
    with pytest.raises(ValueError):
        result.lift('on', 'on')
    with pytest.raises(ValueError):
        result.credible_interval('on', level=1.5)
