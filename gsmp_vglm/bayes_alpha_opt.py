"""
Module for Bayesian optimization of the alpha parameter in GSMP GLM models.
"""

import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

from .grad_desc_sem_functions import GSMP_GLM  


def case_1_2_log_like(alpha_val: float, X_data: np.ndarray, N_data: np.ndarray, cov_data: np.ndarray, eta_vector: np.ndarray) -> float:
    """Compute the log-likelihood for given alpha and eta values."""
    alpha_sum = 0
    for i in range(len(N_data)):
        N_i = int(N_data[i])
        for j in range(N_i):
            alpha_sum += np.log(j * alpha_val + 1)

    left_sum = np.dot(N_data, np.dot(cov_data, eta_vector))
    right_sum = np.sum((1 / alpha_val + N_data) * np.log(1 + alpha_val * X_data * np.exp(np.dot(cov_data, eta_vector))))

    return alpha_sum + left_sum - right_sum


def objective_function(alpha: list, data: dict, learning_rate: float, sgd_itr: int, init_multiplier: float) -> float:
    """Objective function for Bayesian optimization: negative log-likelihood."""
    model = GSMP_GLM()
    
    eta_est, beta_est, _, _ = model.fit(
        data['X'].values, data['N'].values, data['W'], data['Z'],
        alpha[0], learning_rate=learning_rate, num_iterations=sgd_itr, init_multiplier=init_multiplier
    )

    loglike = case_1_2_log_like(
        alpha[0],
        data['X'].values,
        data['N'].values,
        data['W'],
        eta_est
    )

    return -loglike  # Minimize negative log-likelihood


def bayesian_optimization(data: dict,
                           initial_alpha: float,
                           learning_rate: float,
                           sgd_itr: int,
                           init_multiplier: float,
                           n_calls: int = 50,
                           alpha_bounds: tuple = (1e-5, 1.5)):
    """Perform Bayesian optimization to find the optimal alpha value."""

    alpha_range = [alpha_bounds]

    result = gp_minimize(
        func=lambda alpha: objective_function(alpha, data, learning_rate, sgd_itr, init_multiplier),
        dimensions=alpha_range,
        x0=[initial_alpha],
        acq_func='EI',
        n_calls=n_calls,
        random_state=1234
    )

    return result


def plot_optimization_convergence(result) -> None:
    """Plot convergence of Bayesian optimization."""
    plot_convergence(result)
    plt.title("Bayesian Optimization Convergence for Alpha")
    plt.xlabel("Number of Calls")
    plt.ylabel("Negative Log-Likelihood")
    plt.show()
