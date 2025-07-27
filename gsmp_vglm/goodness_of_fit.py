"""
Module for performing Goodness-of-Fit for GSMP-VGLM model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gamma


class GSMPGoodnessOfFit:
    def __init__(self, eta_vec, beta_vec, X, Y, W, Z, alpha, n_sims=10000):
        self.eta_vec = eta_vec
        self.beta_vec = beta_vec
        self.X = X
        self.Y = Y
        self.W = W
        self.Z = Z
        self.alpha = alpha
        self.n_sims = n_sims

        self.lamb_vec = np.exp(np.dot(self.W, self.eta_vec))
        self.p_vec = np.exp(np.dot(self.Z, self.beta_vec)) / (1 + np.exp(np.dot(self.Z, self.beta_vec)))

    def cdf_X(self, xs):
        return 1 - (1 + self.alpha * self.lamb_vec * xs * self.p_vec) ** (-1 / self.alpha)

    def cdf_Y(self, ys):
        n_obs = len(ys)
        tj = gamma.rvs(1 / self.alpha, scale=1, size=(n_obs, self.n_sims))
        exp_part = 1 - np.exp(-1 * self.alpha * self.lamb_vec[:, None] * ys[:, None] * tj)
        ratio = (self.p_vec[:, None] * exp_part) / (1 - (1 - self.p_vec[:, None]) * exp_part)
        return np.mean(ratio, axis = 1)

    def run_analysis(self, plot=True):
        x_unif_vec = self.cdf_X(self.X)
        y_unif_vec = self.cdf_Y(self.Y)

        x_ks_stat, x_ks_p = stats.kstest(x_unif_vec, stats.uniform.cdf)
        y_ks_stat, y_ks_p = stats.kstest(y_unif_vec, stats.uniform.cdf)

        if plot:
            fig, axs = plt.subplots(1, 2, figsize = (12, 5))

            stats.probplot(x_unif_vec, dist = 'uniform', plot = axs[0])
            axs[0].set_title('Uniform(0,1) QQ-Plot for Magnitude (X)')

            stats.probplot(y_unif_vec, dist = 'uniform', plot = axs[1])
            axs[1].set_title('Uniform(0,1) QQ-Plot for Maximum (Y)')

            plt.tight_layout()
            plt.show()

        return {
            'x_unif_vec': x_unif_vec,
            'y_unif_vec': y_unif_vec,
            'x_ks_test': {'statistic': x_ks_stat, 'p_value': x_ks_p},
            'y_ks_test': {'statistic': y_ks_stat, 'p_value': y_ks_p},
        }
