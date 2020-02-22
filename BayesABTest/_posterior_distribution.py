"""Dec file for a posterior distribution class."""


class posterior_distribution:
    """Handle all of the posterior information needed for one variant.

    This mostly just serves as an information container.
    """

    def __init__(self, variant_name, color, dist_type):
        """Init the posterior class."""
        self.__dist_type = dist_type
        self.__variant_name = variant_name
        self.__color = color
        self.__metric_agg = None
        self.__bucketed = None
        self.__variance = None
        self.__likelihood_sample = None

        self.__posterior_mean = None
        self.__posterior_var = None
        self.__posterior_sample_name = None
        self.__posterior_sample = None

    def get_dist_type(self):
        """Get the distribution type of the variant."""
        return self.__dist_type

    def get_variant_name(self):
        """Get variant name."""
        return self.__variant_name

    def get_color(self):
        """Get variants plot color."""
        return self.__color

    def get_metric_agg(self):
        """Get the aggregated empirical metric value for the variant."""
        assert self.__metric_agg is not None
        assert self.__dist_type in ['beta', 'poisson']
        return self.__metric_agg

    def set_metric_agg(self, metric_agg):
        """Set the aggregated empirical metric value for the variant."""
        assert self.__dist_type in ['beta', 'poisson']
        self.__metric_agg = metric_agg

    def get_bucketed(self):
        """Get the number bucketed for the test for the variant."""
        assert self.__bucketed is not None
        assert self.__dist_type in ['beta', 'poisson']
        return self.__bucketed

    def set_bucketed(self, bucketed):
        """Set the number bucketed for the test for the variant."""
        assert self.__dist_type in ['beta', 'poisson']
        self.__bucketed = bucketed

    def get_variance(self):
        """Get the variance for the test for the variant."""
        assert self.__variance is not None
        assert self.__dist_type in ['poisson', 'log-normal', 'normal']
        return self.__variance

    def set_variance(self, variance):
        """Set the variance for the test for the variant."""
        assert self.__dist_type in ['poisson', 'log-normal', 'normal']
        self.__variance = variance

    def get_likelihood_sample(self):
        """Get the likelihood_sample for the variant."""
        assert self.__likelihood_sample is not None
        return self.__likelihood_sample

    def set_likelihood_sample(self, likelihood_sample):
        """Set the likelihood_sample for the variant."""
        assert isinstance(likelihood_sample, list)
        self.__likelihood_sample = likelihood_sample

    def get_posterior_mean(self):
        """Get the posterior_mean for the variant."""
        assert self.__posterior_mean is not None
        assert self.__dist_type in ['log-normal', 'normal']
        return self.__posterior_mean

    def set_posterior_mean(self, posterior_mean):
        """Set the posterior_mean for the variant."""
        assert self.__dist_type in ['log-normal', 'normal']
        self.__posterior_mean = posterior_mean

    def get_posterior_var(self):
        """Get the posterior_var for the variant."""
        assert self.__posterior_var is not None
        assert self.__dist_type in ['log-normal', 'normal']
        return self.__posterior_var

    def set_posterior_var(self, posterior_var):
        """Set the posterior_var for the variant."""
        assert self.__dist_type in ['log-normal', 'normal']
        self.__posterior_var = posterior_var

    def get_posterior_sample_name(self):
        """Get the posterior_sample_name for the variant."""
        assert self.__posterior_sample_name is not None
        assert self.__dist_type in ['log-normal', 'normal']
        return self.__posterior_sample_name

    def set_posterior_sample_name(self, posterior_sample_name):
        """Set the posterior_sample_name for the variant."""
        assert self.__dist_type in ['log-normal', 'normal']
        self.__posterior_sample_name = posterior_sample_name

    def get_posterior_sample(self):
        """Get the posterior_sample for the variant."""
        assert self.__posterior_sample is not None
        return self.__posterior_sample

    def set_posterior_sample(self, posterior_sample):
        """Set the posterior_sample for the variant."""
        self.__posterior_sample = posterior_sample
