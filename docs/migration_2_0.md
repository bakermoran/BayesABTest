# Migrating from 1.x to 2.0

2.0 is a breaking release. The API was redesigned, and several results were wrong in 1.x and changed.

## API changes

Before:

```python
from BayesABTest import ab_test_model

model = ab_test_model(df, metric='conv', prior_function='beta',
                      prior_info='informed', control_bucket_name='off',
                      bucket_col_name='bucket', samples=10000,
                      confidence_level=.05, compare_variants=True,
                      prior_scale_factor=4, debug=True)
model.fit()
model.plot()
model.lift['on']['off']
model.loss_function()
```

After:

```python
from BayesABTest import ABTest, priors

result = ABTest('conv', 'beta', control='off', group='bucket',
                prior=priors.Empirical(scale=4), n_samples=10000,
                random_state=0).fit(df)
result.plot(compare_variants=True, ci=0.95)
result.lift('on')
result.loss('on')
```

| 1.x | 2.0 |
| --- | --- |
| `ab_test_model(raw_data, ...)` then `.fit()` | `ABTest(...).fit(raw_data)` returns a `Result` |
| `prior_function='beta'` | `family='beta'` |
| `prior_info='uninformed'` | `prior=priors.Uninformed()` |
| `prior_info='informed'`, `prior_scale_factor=n` | `prior=priors.Empirical(scale=n)` (the default) |
| `prior_info='specified'`, `prior_parameters={'alpha': a, 'beta': b}` | `prior=priors.Beta(a, b)` (`Gamma` for poisson) |
| `prior_parameters={'mean': m, 'var': v}` (poisson) | `prior=priors.Gamma.from_mean_var(m, v)`. The variance is no longer multiplied by the scale factor |
| `prior_parameters={'mean': m, 'var': v}` (normal, log-normal) | `priors.Normal(m, v)` / `priors.LogNormal(m, v)` (log scale) |
| `control_bucket_name`, `bucket_col_name` | `control`, `group` |
| `samples` (1,000 to 50,000) | `n_samples` (at least 1,000, no upper limit) |
| `confidence_level=.05` (it was really alpha) | `ci=0.95` on `plot()` and `plot_ecdf()` |
| `compare_variants=True` in the constructor | `plot(compare_variants=True)`. Any pair can be compared with `result.lift(a, b)` |
| `debug=True` | removed. Nothing is printed |
| no way to seed | `random_state=` |
| `model.lift[a][b]`, `model.ecdf[a][b]` | `result.lift(a, b)`, `result.ecdf(a, b)` |
| `model.posteriors[x].get_posterior_sample()` (a list) | `result.samples[x]` (a column of a DataFrame) |
| `model.posteriors[x].get_posterior_mean()` | `result.posteriors[x].params['mean']` |
| `model.plot(lift_plot_flag=...)` | `result.plot(lift=...)`, returns a `Figure` |
| `model.plot_positive_lift(a, b)` | `result.plot_lift(a, b)`, returns an `Axes` |
| `model.plot_ecdf(a, b)`, `plot_posteriors(variants=[...])` | same names on `Result`, with an `ax=` argument |
| `model.loss_function()` (printed, one variant) | `result.loss(variant)` returns a `LossResult`, for any variant |
| the limit of 10 variants | removed |

New: `ABTest.fit_counts(successes, trials)` for aggregated beta data, `Result.prob_better`, `Result.credible_interval` and `Result.summary`.

The distribution explorer functions keep their names. They now return the frozen `scipy.stats` distribution and take an `ax=` keyword. Importing `ab_test_dist_explorer` with `*` no longer re-exports numpy, seaborn and matplotlib.

## Unchanged

The log-normal posterior still describes exp(mu), the median, so `lift` for `log-normal` is a ratio of medians, not of means. The normal and log-normal models still treat the sample variance as known. These are documented approximations.
