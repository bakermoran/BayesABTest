# Bayesian AB Test Report Builder

![Build Status](https://github.com/bakermoran/BayesABTest/workflows/Build%20Status/badge.svg?branch=main)
[![PyPI version](https://badge.fury.io/py/BayesABTest.svg)](https://badge.fury.io/py/BayesABTest)

> **Baker Moran**

This allows AB testers to create a standard report for test results with python. Handles multiple variant tests, with a variety of prior function types. This is meant to be an abstraction from the nitty gritty of the details of Bayesian statistics. That said, some knowledge about which prior function to use etc. will be helpful for the user.

### Example Output for a conversion rate test

![Conversion rate example report](https://raw.githubusercontent.com/bakermoran/BayesABTest/main/docs/package_documentation/img/one_var_conversion.png)

## Installation

- Install via pip or [uv](https://docs.astral.sh/uv/) (requires Python 3.12+)

  ```bash
  pip install BayesABTest
  # or
  uv add BayesABTest
  ```

  OR

- Download files from [PyPi](https://pypi.org/project/BayesABTest/#files) and install

## Quick start

```python
import pandas as pd
from matplotlib import pyplot as plt
from BayesABTest import ABTest

df = pd.read_csv('results.csv')   # a metric column and a bucket column

result = ABTest('conversion', 'beta', control='off', random_state=0).fit(df)

result.prob_better('on')          # probability the variant beats the control
result.summary()                  # posterior mean, interval and lift per bucket
result.plot()                     # the standard report
plt.show()
```

`family` is `'beta'` (conversion rates), `'log-normal'`, `'normal'` or `'poisson'`. See the [ABTest docs](https://github.com/bakermoran/BayesABTest/blob/main/docs/package_documentation/ab_test.md) for priors, loss and plotting options.

## Development

```bash
uv sync                # create .venv and install runtime + dev dependencies
uv run pytest -n 4     # run tests
uv run ruff check .    # lint
uv build               # build sdist and wheel
```

## Package Functions and Classes

### [ABTest](https://github.com/bakermoran/BayesABTest/blob/main/docs/package_documentation/ab_test.md)

Class implementing out of the box AB testing functionality. Simple, easy to use AB testing with many different prior function types, all in one clean interface.

### [ab_test_dist_explorer](https://github.com/bakermoran/BayesABTest/blob/main/docs/package_documentation/ab_test_dist_explorer.md)

Functions allowing a user to explore different distributions with simple to user interface. Allows a user to visually learn about bayesian statistics, and inform proper prior function choice when doing Bayesian AB testing.

## Upgrading from 1.x

2.0 changes the API and fixes several math errors. See [the migration guide](https://github.com/bakermoran/BayesABTest/blob/main/docs/migration_2_0.md).

## Appendix

### Learning

For a documentation explaining and motivating the use of Bayesian statistics to evaluate A/B tests, see [documentation](https://github.com/bakermoran/BayesABTest/blob/main/docs/besyian_ab_testing/Bayesian_AB_Testing_explainer.md)

### Acknowledgements

There is a lot of documentation out there about a Bayesian framework of A/B testing. Some of the specific articles are listed below. Most of the work I came across was written in R, and I set out to create a Python implementation. The visuals were inspired by a standard template we use at Root, first written by <https://github.com/zachurchill-root>.

Articles Reference:

- <https://www.countbayesie.com/blog/2015/4/25/bayesian-ab-testing>
- <https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5>
- <https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf>

### Version History

- `0.1.0-prealpha` - _12/02/2019_
- `1.0.0-alpha` - _12/27/2019_
- `1.0.1-alpha` - _01/02/2020_
- `1.0.2-alpha` - _06/17/2020_
- `1.0.3-alpha` - _06/17/2020_
- `1.0.4-alpha` - _06/22/2020_
- `1.0.5-alpha` - _06/22/2020_
- `1.0.6-alpha` - _06/22/2020_
- `1.0.7-alpha` - _07/09/2021_
- `1.0.8-alpha` - _03/29/2022_
- `1.0.9-alpha` - _03/29/2022_
- `2.0.0` - API redesign and math fixes, see the [migration guide](https://github.com/bakermoran/BayesABTest/blob/main/docs/migration_2_0.md)
