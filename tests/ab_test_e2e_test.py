"""End to end: generate data, fit every family and prior type, plot."""

import data_helpers as dh
import pytest

from BayesABTest import ABTest, priors

N = 1000


def _check(result, n_variants):
    assert len(result.variants) == n_variants
    assert result.samples.shape == (N, n_variants + 1)
    assert not result.samples.isna().any().any()
    assert len(result.summary()) == n_variants + 1
    result.plot(compare_variants=1 < n_variants <= 3)


@pytest.mark.parametrize('rates, names', [
    ([.27, .3], ['off', 'on']),
    ([.27, .23], ['control', 'variant']),
    ([.2, .3, .4], ['control', 'variant1', 'variant2']),
])
@pytest.mark.parametrize('prior', [priors.Uninformed(), priors.Empirical(),
                                   priors.Beta(22, 78)])
def test_conversion(rates, names, prior):
    data = dh.create_conversion_data(rates, names, metric_name='conv')
    result = ABTest('conv', 'beta', control=names[0], prior=prior,
                    n_samples=N, random_state=0).fit(data)
    _check(result, len(rates) - 1)


@pytest.mark.parametrize('family, log', [('log-normal', True),
                                         ('normal', False)])
@pytest.mark.parametrize('means', [[600, 700], [600, 610, 615],
                                   [600, 610, 615, 620, 625]])
@pytest.mark.parametrize('prior', [priors.Uninformed(), priors.Empirical()])
def test_continuous(family, log, means, prior):
    names = ['control'] + [f'variant_{i}' for i in range(1, len(means))]
    data = dh.create_continuous_data(means, [1.5] * len(means), names,
                                     metric_name='premium', log=log)
    result = ABTest('premium', family, control='control', prior=prior,
                    n_samples=N, random_state=0).fit(data)
    _check(result, len(means) - 1)


@pytest.mark.parametrize('family, prior', [
    ('log-normal', priors.LogNormal(6.4, 1.5)),
    ('normal', priors.Normal(650, 30000)),
])
def test_specified_continuous_prior(family, prior):
    data = dh.create_continuous_data(
        [600, 610, 615], [30, 30, 30], ['control', 'variant_1', 'variant_2'],
        metric_name='premium', log=family == 'log-normal')
    result = ABTest('premium', family, control='control', prior=prior,
                    n_samples=N, random_state=0).fit(data)
    assert result.prior is prior
    _check(result, 2)


@pytest.mark.parametrize('prior', [priors.Empirical(),
                                   priors.Gamma(8, 2),
                                   priors.Gamma.from_mean_var(15, 3)])
@pytest.mark.parametrize('rates', [[10, 20, 25], [25, 50, 25]])
def test_poisson(rates, prior):
    names = ['control', 'variant1', 'variant2']
    data = dh.create_poisson_data(rates, names, metric_name='visits',
                                  sample_length=3000)
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    result = ABTest('visits', 'poisson', control='control', prior=prior,
                    n_samples=N, random_state=0).fit(data)
    _check(result, 2)
