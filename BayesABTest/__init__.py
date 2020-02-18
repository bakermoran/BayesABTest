"""
BayesABTest python package.

A simple to use interface for Bayesian
hypothesis testing. Meant to provide analysts and data scientists with an
abstracted way to quickly test a hypothesis. Provides a uniform reporting
chart for test results.
"""
# pylint: disable=unused-import

from .ab_test_model import ab_test_model  # noqa

# to do: log normal and beta 'get prior' and 'get posteriors' should have the
# same interface, but they dont
# they're private right now so it doesnt matter too much for the time being
# to do: get log normal uninformed prior right
# to do: on the cdf plot, dont despine the left (when i try this it
# applies to all of them)
# to do: allow a user to specify the prior
# to do: separate posterior generation from the inputs.
# make it just a function that takes data
# to do: get everything 79 chars or less on one line
# to do: add unit tests
# to do: add ability to not sample, but run with seeded random numbers
# (scipy or numpy)
# to do: allow someone to input a lot of data, but select out two
# variants to plot
# ex: data contains var1, var2, var3, but want to only test var1 and var3
# for some reason, this doesnt work:
# self.raw_data =
# raw_data.loc[raw_data[bucket_col_name].isin(variant_bucket_names)]
# or else do this for now: make sure number of buckets in dataframe match
# number of buckets in variant_bucket_names
# to do: debug should actually output diagnostic info of the model
# to do: add public interface to the single graphing functions,
# with args for variant names
# to do: lift_plot_flag and compare_variants should be an
# input to the create report function

# TO DO: make the inputs part of a class that is a model:
# so like, there is a class with just what can be used for the prior
# class beta, lognormal, normal etc.
