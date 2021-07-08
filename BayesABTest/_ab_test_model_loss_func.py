"""Loss function functionality for ab_test_model."""

import numpy as np
import scipy.special as sc
# pylint: disable=no-member


class _ab_test_loss_functions:
    """Provides loss functions to the greated at_test_model class.

    Raises:
        ValueError: only beta prior is supported
        ValueError: loss type must be absolute or percent
        RuntimeError: only 1 variant is supported
    """

    def _h(self, a, b, c, d):
        """Calculate the probablty that a Beta distributions is greater.

        Returns:
            float -- probabilty that Beta(a,b) is greater than Beta(c,d)
        """
        sum = 0
        for j in range(0, c):
            w = sc.betaln(a + j, b + d)
            x = np.log(d + j)
            y = sc.betaln(1 + j, d)
            z = sc.betaln(a, b)
            sum += np.exp(w - x - y - z)
        print(sum)
        return 1 - sum

    def _loss_beta(self, a, b, c, d, loss_type):
        """Calc loss function value of two dists Beta_1(a,b) and Beta_2(c,d).

        Arguments:
            a {int} -- Beta_1 alpha param
            b {int} -- Beta_1 beta param
            c {int} -- Beta_2 alpha param
            d {int} -- Beta_2 beta param
            loss_type {string} -- 'absolute' or 'percent', is the type of loss
                value returned.

        Returns:
            float -- the loss function value for Beta_1 over Beta_2
        """
        if loss_type == 'absolute':
            x1 = sc.betaln(a + 1, b)
            y1 = np.log(self._h(a + 1, b, c, d))
            z1 = sc.betaln(a, b)

            x2 = sc.betaln(c + 1, d)
            y2 = np.log(self._h(a, b, c + 1, d))
            z2 = sc.betaln(c, d)

            return np.exp(x1 + y1 - z1) - np.exp(x2 + y2 - z2)

        elif loss_type == 'percent':
            prob_1 = self._h(a, b, c, d)

            x = sc.betaln(a - 1, b)
            y = sc.betaln(a, b)
            z = sc.betaln(c + 1, d)
            w = sc.betaln(c, d)
            prob_2 = np.log(self._h(a - 1, b, c + 1, d))

            return prob_1 - np.exp(x - y + z - w + prob_2)

    def _loss_gamma(self, a, b, c, d):
        """Calc loss function value of two dists Gamma_1(a,b) and Gamma_2(c,d).

        Arguments:
            a {int} -- Gamma_1 alpha param
            b {int} -- Gamma_1 beta param
            c {int} -- Gamma_2 alpha param
            d {int} -- Gamma_2 beta param
            loss_type {string} -- 'absolute' or 'percent', is the type of loss
                value returned.

        Returns:
            float -- the loss function value for Gamma_1 over Gamma_2
        """
        x1 = sc.gammaln(a+1)
        y1 = np.log(self._h(a+1, b, c, d))
        z1 = np.log(b)
        w1 = sc.gammaln(a)

        x2 = sc.gammaln(c+1)
        y2 = np.log(self._h(a, b, c+1, d))
        z2 = np.log(d)
        w2 = sc.gammaln(c)

        return np.exp(x1 + y1 - z1 - w1) - np.exp(x2 + y2 - z2 - w2)

    def _loss_normal(self, sample1, sample2):
        """Calc loss function value of two dists Norm_1(a,b) and Norm_2(c,d).

        Arguments:
            sample1 {list} -- Norm_1 samples
            sample2 {list} -- Norm_2 samples
            loss_type {string} -- 'absolute' or 'percent', is the type of loss
                value returned.

        Returns:
            float -- the loss function value for Norm_1 over Norm_2
        """
        diffs = []
        for x, y in zip(sample1, sample2):
            diffs.append(max(x-y), 0)
        return np.mean(diffs)

    def _stop_test(self, risk_A, risk_B, epsilon):
        if risk_A <= epsilon or risk_B <= epsilon:
            if risk_A <= epsilon:
                return True, self.control_bucket_name
            else:
                return True, self.variant_bucket_names[0]
        else:
            return False, None

    def loss_function(self, loss_type='absolute', epsilon=0.0001):
        """Control to calculate the loss function for the variants.

        Currently only conversion 1-variant is supported.
        """
        if loss_type not in ['absolute', 'percent']:
            raise ValueError('loss_type must be either absolute or percent')
        if self.prior_function != 'beta':
            raise ValueError('prior_function must be beta')
        if len(self.variant_bucket_names) != 1:
            raise RuntimeError('only one variant is supported')

        raw_data_agg = self.raw_data.groupby(self.bucket_col_name).sum()
        raw_data_agg['bucketed'] = self.raw_data.groupby(
                                     self.bucket_col_name).count()[self.metric]

        # to do: this is not very clean
        if self.prior_info == 'uninformed':
            prior_alpha = 1
            prior_beta = 1
        else:
            prior_converts = raw_data_agg.loc[
                                         self.control_bucket_name][self.metric]
            prior_non_converts = raw_data_agg.loc[
                                         self.control_bucket_name][
                                             'bucketed'] - prior_converts
            prior_alpha = prior_converts / self.prior_scale_factor
            prior_beta = prior_non_converts / self.prior_scale_factor

        control_alpha_likelihood = raw_data_agg.loc[
                                    self.control_bucket_name][
                                    self.metric]
        control_beta_likelihood = raw_data_agg.loc[
                                    self.control_bucket_name][
                                    'bucketed'] - control_alpha_likelihood

        variant_alpha_likelihood = raw_data_agg.loc[
                                    self.variant_bucket_names[0]][self.metric]
        variant_beta_likelihood = raw_data_agg.loc[
                                    self.variant_bucket_names[0]][
                                    'bucketed'] - variant_alpha_likelihood

        a = control_alpha_likelihood + prior_alpha
        b = control_beta_likelihood + prior_beta

        c = variant_alpha_likelihood + prior_alpha
        d = variant_beta_likelihood + prior_beta

        A_risk = self._loss_beta(c, d, a, b, loss_type=loss_type)
        B_risk = self._loss_beta(a, b, c, d, loss_type=loss_type)
        stop, winner = self._stop_test(A_risk, B_risk, epsilon=epsilon)

        if stop:
            print('Test can be stopped and',
                  winner, 'can be declared the winner')
        else:
            print('Keep running the test')
        print('risk of choosing', self.control_bucket_name, '=', A_risk)
        print('risk of choosing', self.variant_bucket_names[0], '=', B_risk)
