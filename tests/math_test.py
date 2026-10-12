"""Check the model math against closed forms and Monte Carlo."""
import numpy as np
import pandas as pd
import pytest

from BayesABTest import ABTest, priors
from BayesABTest import _conjugate as cj


def _beta_df(seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'bucket': ['off'] * 1000 + ['on'] * 1000,
        'conv': np.r_[rng.binomial(1, .10, 1000), rng.binomial(1, .12, 1000)]})


def _poisson_df(seed=2, n=500):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'bucket': ['off'] * n + ['on'] * n,
                         'n': np.r_[rng.poisson(3, n), rng.poisson(4, n)]})


# ------------------------------------------------------- pure beta math

@pytest.mark.parametrize('params', [(21, 81, 30, 72),
                                    (20.5, 80.25, 7.5, 31),
                                    (135.5, 669.25, 120.25, 800.5)])
def test_prob_greater_and_losses_match_monte_carlo(params):
    a, b, c, d = params
    rng = np.random.default_rng(0)
    x1, x2 = rng.beta(a, b, 2_000_000), rng.beta(c, d, 2_000_000)
    diff = np.maximum(x1 - x2, 0)
    assert cj.prob_beta_greater(a, b, c, d) == pytest.approx(
        (x1 > x2).mean(), abs=2e-3)
    assert cj.expected_loss_beta(a, b, c, d, 'absolute') == pytest.approx(
        diff.mean(), rel=3e-2)
    assert cj.expected_loss_beta(a, b, c, d, 'percent') == pytest.approx(
        (diff / x2).mean(), rel=3e-2)


def test_percent_loss_needs_alpha_above_one():
    with pytest.raises(ValueError):
        cj.expected_loss_beta(5, 5, 1, 5, 'percent')


def test_conjugate_updates_closed_form():
    assert cj.beta_posterior(2, 3, 10, 40) == (12, 33)
    assert cj.gamma_poisson_posterior(2, 1, 30, 10) == (32, 11)
    x = np.array([1., 2., 3., 4.])
    post_mean, post_var = cj.normal_posterior(0, 100, x)
    assert post_var == pytest.approx(1 / (1 / 100 + 4 / 1.25))
    assert post_mean == pytest.approx(post_var * (2.5 * 4 / 1.25))


# ------------------------------------------------------------- priors

def test_empirical_beta_prior_is_scaled_control_counts():
    df = _beta_df()
    result = ABTest('conv', 'beta').fit(df)
    ctrl = df[df.bucket == 'off'].conv
    assert result.prior.alpha == ctrl.sum() / 4
    assert result.prior.beta == (len(ctrl) - ctrl.sum()) / 4


def test_empirical_beta_prior_needs_both_outcomes():
    df = _beta_df()
    df.loc[df.bucket == 'off', 'conv'] = 0
    with pytest.raises(ValueError):
        ABTest('conv', 'beta').fit(df)


def test_gamma_from_mean_var_moment_matches():
    prior = priors.Gamma.from_mean_var(3, 8)
    assert prior.alpha == pytest.approx(9 / 8)
    assert prior.beta == pytest.approx(3 / 8)


def test_empirical_poisson_prior_is_moment_matched_to_control():
    df = _poisson_df()
    result = ABTest('n', 'poisson').fit(df)
    ctrl = df[df.bucket == 'off'].n
    sigma2 = ctrl.var() * 4
    assert result.prior.alpha == pytest.approx(ctrl.mean() ** 2 / sigma2)
    assert result.prior.beta == pytest.approx(ctrl.mean() / sigma2)


def test_empirical_normal_prior_uses_scaled_control_moments():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({'bucket': ['off'] * 300 + ['on'] * 300,
                       'v': np.r_[rng.normal(10, 2, 300),
                                  rng.normal(11, 2, 300)]})
    result = ABTest('v', 'normal', prior=priors.Empirical(scale=2)).fit(df)
    ctrl = df[df.bucket == 'off'].v.values
    assert result.prior.mean == pytest.approx(ctrl.mean())
    assert result.prior.var == pytest.approx(ctrl.var() * 2)


# --------------------------------------------------------- posteriors

@pytest.mark.parametrize('prior', [priors.Uninformed(), priors.Empirical(),
                                   priors.Beta(2, 8)])
def test_beta_posterior_matches_closed_form(prior):
    df = _beta_df()
    result = ABTest('conv', 'beta', prior=prior, n_samples=50000,
                    random_state=0).fit(df)
    k = df[df.bucket == 'off'].conv.sum()
    n_ctrl = (df.bucket == 'off').sum()
    prior_a, prior_b = {priors.Uninformed: (1, 1),
                        priors.Empirical: (k / 4, (n_ctrl - k) / 4),
                        priors.Beta: (2, 8)}[type(prior)]
    for name in ['off', 'on']:
        sub = df[df.bucket == name].conv
        a = prior_a + sub.sum()
        b = prior_b + len(sub) - sub.sum()
        assert result.posteriors[name].params == {'alpha': a, 'beta': b}
        s = result.samples[name]
        assert s.mean() == pytest.approx(a / (a + b), abs=2e-3)
        assert s.var() == pytest.approx(
            a * b / ((a + b) ** 2 * (a + b + 1)), rel=0.05)


def test_poisson_posterior_is_conjugate_gamma():
    df = _poisson_df()
    result = ABTest('n', 'poisson', prior=priors.Gamma(2, 1),
                    n_samples=50000, random_state=0).fit(df)
    for name in ['off', 'on']:
        total, n = df[df.bucket == name].n.sum(), 500
        assert result.posteriors[name].params == {'alpha': 2 + total,
                                                  'beta': 1 + n}
        s = result.samples[name]
        assert s.mean() == pytest.approx((2 + total) / (1 + n), rel=0.01)
        assert s.std() == pytest.approx(np.sqrt(2 + total) / (1 + n),
                                        rel=0.03)


def test_normal_posterior_matches_conjugate_update():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({'bucket': ['off'] * 400 + ['on'] * 400,
                       'v': np.r_[rng.normal(10, 2, 400),
                                  rng.normal(11, 2, 400)]})
    result = ABTest('v', 'normal', prior=priors.Normal(0, 100),
                    n_samples=50000, random_state=0).fit(df)
    for name in ['off', 'on']:
        x = df[df.bucket == name].v.values
        post_var = 1 / (1 / 100 + len(x) / x.var())
        post_mean = post_var * (x.mean() * len(x) / x.var())
        params = result.posteriors[name].params
        assert params['var'] == pytest.approx(post_var)
        assert params['mean'] == pytest.approx(post_mean)
        s = result.samples[name]
        assert s.mean() == pytest.approx(post_mean, abs=0.01)
        assert s.std() == pytest.approx(np.sqrt(post_var), rel=0.03)


def test_lognormal_posterior_is_conjugate_on_log_scale():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({'bucket': ['off'] * 400 + ['on'] * 400,
                       'v': np.r_[rng.lognormal(1.0, 0.5, 400),
                                  rng.lognormal(1.2, 0.5, 400)]})
    result = ABTest('v', 'log-normal', prior=priors.LogNormal(0, 100),
                    n_samples=50000, random_state=0).fit(df)
    for name in ['off', 'on']:
        logx = np.log(df[df.bucket == name].v.values)
        post_var = 1 / (1 / 100 + len(logx) / logx.var())
        post_mean = post_var * (logx.mean() * len(logx) / logx.var())
        s = np.log(result.samples[name])
        assert s.mean() == pytest.approx(post_mean, abs=0.01)
        assert s.std() == pytest.approx(np.sqrt(post_var), rel=0.03)


# --------------------------------------------------------- lift / ecdf

def _three_bucket_result():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        'bucket': ['off'] * 500 + ['v1'] * 500 + ['v2'] * 500,
        'n': np.r_[rng.poisson(3, 500), rng.poisson(3.3, 500),
                   rng.poisson(3.6, 500)]})
    return ABTest('n', 'poisson', random_state=0).fit(df)


def test_lift_is_numerator_over_denominator_minus_one():
    result = _three_bucket_result()
    s = result.samples
    np.testing.assert_allclose(result.lift('v1'), s['v1'] / s['off'] - 1)
    np.testing.assert_allclose(result.lift('v2', 'v1'), s['v2'] / s['v1'] - 1)
    np.testing.assert_allclose(result.lift('off', 'v2'), s['off'] / s['v2'] - 1)


def test_prob_better_and_credible_interval():
    result = _three_bucket_result()
    lift = result.lift('v2')
    assert result.prob_better('v2') == (lift > 0).mean()
    low, high = result.credible_interval('v2', level=0.9)
    assert low == pytest.approx(np.quantile(lift, 0.05))
    assert high == pytest.approx(np.quantile(lift, 0.95))
    assert low < np.median(lift) < high


def test_ecdf_is_a_valid_cdf_of_the_lift():
    result = _three_bucket_result()
    lift = result.lift('v1', 'v2')
    x, y = result.ecdf('v1', 'v2')
    assert np.all(np.diff(x) > 0)
    assert np.all(np.diff(y) >= 0)
    assert y[-1] == 1.0
    assert x[0] == lift.min() and x[-1] == lift.max()
    for i in np.linspace(0, len(x) - 1, 25).astype(int):
        assert y[i] == pytest.approx((lift <= x[i]).mean())


def test_summary_columns_and_control_row():
    summary = _three_bucket_result().summary()
    assert list(summary.index) == ['off', 'v1', 'v2']
    assert summary.loc['off', ['median_lift', 'prob_better']].isna().all()
    assert (summary.ci_low < summary['mean']).all()
    assert (summary['mean'] < summary.ci_high).all()
    assert summary.loc['v2', 'prob_better'] > summary.loc['v1', 'prob_better']


# ---------------------------------------------------------------- loss

@pytest.mark.parametrize('loss_type', ['absolute', 'percent'])
def test_loss_risks_match_posterior_samples(loss_type):
    df = _beta_df()
    out = ABTest('conv', 'beta', prior=priors.Uninformed()).fit(df).loss(
        'on', loss_type=loss_type)
    rng = np.random.default_rng(0)
    draws = {}
    for name in ['off', 'on']:
        sub = df[df.bucket == name].conv
        draws[name] = rng.beta(1 + sub.sum(), 1 + len(sub) - sub.sum(),
                               2_000_000)
    ctrl, var = draws['off'], draws['on']
    if loss_type == 'absolute':
        risk_ctrl = np.maximum(var - ctrl, 0).mean()
        risk_var = np.maximum(ctrl - var, 0).mean()
    else:
        risk_ctrl = (np.maximum(var - ctrl, 0) / ctrl).mean()
        risk_var = (np.maximum(ctrl - var, 0) / var).mean()
    assert out.risk_control == pytest.approx(risk_ctrl, rel=0.03)
    assert out.risk_variant == pytest.approx(risk_var, rel=0.03)


def test_loss_works_with_an_empirical_prior():
    out = ABTest('conv', 'beta').fit(_beta_df()).loss('on')
    assert 0 < out.risk_variant < out.risk_control


def test_loss_declares_clear_winner():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({'bucket': ['off'] * 3000 + ['on'] * 3000,
                       'conv': np.r_[rng.binomial(1, .10, 3000),
                                     rng.binomial(1, .20, 3000)]})
    out = ABTest('conv', 'beta', prior=priors.Uninformed()).fit(df).loss('on')
    assert out.stop is True
    assert out.winner == 'on'
    assert out.risk_variant < out.risk_control
    assert 'can be declared the winner' in str(out)


def test_loss_keeps_running_when_indistinguishable():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({'bucket': ['off'] * 200 + ['on'] * 200,
                       'conv': rng.binomial(1, .10, 400)})
    out = ABTest('conv', 'beta', prior=priors.Uninformed()).fit(df).loss('on')
    assert out.stop is False
    assert out.winner is None
    assert 'Keep running' in str(out)


def test_loss_epsilon_picks_the_side_below_it():
    result = ABTest('conv', 'beta', prior=priors.Uninformed()).fit(_beta_df())
    risk = result.loss('on')
    # a threshold between the two risks selects the lower-risk side
    mid = (risk.risk_control + risk.risk_variant) / 2
    out = result.loss('on', epsilon=mid)
    assert out.stop and out.winner == 'on'


def test_loss_is_beta_only_and_validates_variant():
    with pytest.raises(ValueError):
        ABTest('n', 'poisson').fit(_poisson_df()).loss('on')
    result = ABTest('conv', 'beta').fit(_beta_df())
    with pytest.raises(ValueError):
        result.loss('off')
    with pytest.raises(ValueError):
        result.loss('on', loss_type='bogus')
