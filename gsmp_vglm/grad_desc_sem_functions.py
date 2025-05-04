"""
Gradient descent methods for estimating parameters in TETLG and GSMP models.
"""

import numpy as np
from scipy.stats import uniform, norm, geom, expon
from scipy.optimize import minimize, fsolve
from numpy.linalg import matrix_rank
from typing import Tuple, Optional, Union, List, Dict, Any


class BaseEstimator:
    """Base class for all estimators."""
    
    def __init__(self):
        """Initialize the base estimator."""
        pass

    def fit(self, data: Dict[str, np.ndarray]) -> Tuple:
        """Fit the model to data.
        
        Args:
            data: Dictionary of data arrays
            
        Returns:
            Tuple of estimated parameters
        """
        raise NotImplementedError("fit method must be implemented by subclasses")

    def predict(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Make predictions using the fitted model.
        
        Args:
            data: Dictionary of input data arrays
            
        Returns:
            Dictionary of predicted values
        """
        raise NotImplementedError("predict method must be implemented by subclasses")


class TETLG_GLM(BaseEstimator):
    """Estimator for the TETLG GLM model."""
    
    def __init__(self):
        """Initialize the TETLG GLM estimator."""
        super().__init__()
        self.eta_est = None
        self.beta_est = None
        self.eta_cost = None
        self.beta_cost = None
        
    def mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate mean squared error between two arrays.
        
        Args:
            X: First array
            Y: Second array
            
        Returns:
            Mean squared error
        """
        n = len(X)
        return 1/n * np.sum((X * Y) ** 2)
    
    def gradient_g(self, eta_hat: np.ndarray, X_data: np.ndarray, 
                  N_data: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """Compute gradient of the log-likelihood with respect to eta parameters.
        
        Args:
            eta_hat: Current eta parameters
            X_data: Magnitude data
            N_data: Duration data
            covariates: Covariate matrix for eta
            
        Returns:
            Gradient vector
        """
        # Using gradient of negative log-likelihood
        exp_terms = np.exp(np.dot(covariates, eta_hat))
        return np.dot(X_data, covariates * exp_terms[:, np.newaxis]) - np.dot(N_data, covariates)

    def gradient_descent_g(self, eta_init: np.ndarray, X_data: np.ndarray, 
                          N_data: np.ndarray, covariates: np.ndarray, 
                          learning_rate: float, num_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform gradient descent to estimate eta parameters.
        
        Args:
            eta_init: Initial eta parameters
            X_data: Magnitude data
            N_data: Duration data
            covariates: Covariate matrix for eta
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of iterations
            
        Returns:
            Tuple containing:
                eta: Estimated eta parameters
                cost: Cost history
        """
        cost = np.array([])
        eta = eta_init.copy()
        for i in range(num_iterations):
            grad = self.gradient_g(eta, X_data, N_data, covariates)
            eta -= learning_rate * grad
            cost = np.append(cost, self.mse(eta, eta_init))
        return eta, cost

    def gradient_h(self, beta_hat: np.ndarray, N_data: np.ndarray, 
                  covariates: np.ndarray) -> np.ndarray:
        """Compute gradient of the log-likelihood with respect to beta parameters.
        
        Args:
            beta_hat: Current beta parameters
            N_data: Duration data
            covariates: Covariate matrix for beta
            
        Returns:
            Gradient vector
        """
        # Uses gradient of negative log-likelihood
        exp_terms = np.exp(np.dot(covariates, beta_hat))
        return np.dot(N_data, (covariates * exp_terms[:, np.newaxis]) / (1 + exp_terms[:, np.newaxis])) - np.sum(covariates, axis=0)

    def gradient_descent_h(self, beta_init: np.ndarray, N_data: np.ndarray, 
                          covariates: np.ndarray, learning_rate: float, 
                          num_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform gradient descent to estimate beta parameters.
        
        Args:
            beta_init: Initial beta parameters
            N_data: Duration data
            covariates: Covariate matrix for beta
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of iterations
            
        Returns:
            Tuple containing:
                beta: Estimated beta parameters
                cost: Cost history
        """
        cost = np.array([])
        beta = beta_init.copy()
        for i in range(num_iterations):
            grad = self.gradient_h(beta, N_data, covariates)
            beta -= learning_rate * grad
            cost = np.append(cost, self.mse(beta, beta_init))
        return beta, cost

    def fit(self, X_data: np.ndarray, N_data: np.ndarray, W_data: np.ndarray, 
           Z_data: np.ndarray, learning_rate: float = 0.00001, 
           num_iterations: int = 2000, init_multiplier: float = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit the TETLG GLM model to data.
        
        Args:
            X_data: Magnitude data
            N_data: Duration data
            W_data: Covariate matrix for eta
            Z_data: Covariate matrix for beta
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of iterations
            init_multiplier: Multiplier for initial parameter values
            
        Returns:
            Tuple containing:
                eta_est: Estimated eta parameters
                beta_est: Estimated beta parameters
                eta_cost: Cost history for eta
                beta_cost: Cost history for beta
        """
        # Initialize parameters randomly
        initial_eta = init_multiplier * norm.rvs(0, 1, size=np.shape(W_data)[1])
        initial_beta = init_multiplier * norm.rvs(0, 1, size=np.shape(Z_data)[1])
        
        # Estimate parameters using gradient descent
        self.eta_est, self.eta_cost = self.gradient_descent_g(
            initial_eta, X_data, N_data, W_data, learning_rate, num_iterations
        )
        self.beta_est, self.beta_cost = self.gradient_descent_h(
            initial_beta, N_data, Z_data, learning_rate, num_iterations
        )

        return self.eta_est, self.beta_est, self.eta_cost, self.beta_cost
    
    def predict(self, W_new: np.ndarray, Z_new: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions for new data.
        
        Args:
            W_new: New covariate matrix for eta
            Z_new: New covariate matrix for beta
            
        Returns:
            Dictionary containing:
                lambda: Predicted lambda values
                p: Predicted p values
        """
        if self.eta_est is None or self.beta_est is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        lambda_pred = np.exp(np.dot(W_new, self.eta_est))
        p_pred = np.exp(np.dot(Z_new, self.beta_est)) / (1 + np.exp(np.dot(Z_new, self.beta_est)))
        
        return {
            'lambda': lambda_pred,
            'p': p_pred
        }

class GSMP_GLM(BaseEstimator):
    """Estimator for the GSMP GLM model."""
    
    def __init__(self):
        """Initialize the GSMP GLM estimator."""
        super().__init__()
        self.eta_est = None
        self.beta_est = None
        self.eta_cost = None
        self.beta_cost = None
        
    def mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate mean squared error between two arrays.
        
        Args:
            X: First array
            Y: Second array
            
        Returns:
            Mean squared error
        """
        n = len(X)
        return 1/n * np.sum((X * Y) ** 2)

    def exact_matrix_solver_eta(self, W_data: np.ndarray, X_data: np.ndarray, 
                               N_data: np.ndarray) -> np.ndarray:
        """Solve for eta parameters using matrix operations.
        
        Args:
            W_data: Covariate matrix for eta
            X_data: Magnitude data
            N_data: Duration data
            
        Returns:
            Estimated eta parameters
        """
        return np.matmul(
            np.linalg.pinv(np.matmul(np.transpose(W_data), W_data)),
            np.matmul(np.transpose(W_data), np.log(N_data / X_data))
        )
    
    def exact_matrix_solver_beta(self, Z_data: np.ndarray, N_data: np.ndarray) -> np.ndarray:
        """Solve for beta parameters using matrix operations.
        
        Args:
            Z_data: Covariate matrix for beta
            N_data: Duration data
            
        Returns:
            Estimated beta parameters
        """
        # Only use observations where N > 1
        mask = N_data > 1
        
        return -np.matmul(
            np.linalg.pinv(np.matmul(np.transpose(Z_data[mask]), Z_data[mask])),
            np.matmul(np.transpose(Z_data[mask]), np.log(N_data[mask] - 1))
        )
    
    def gradient_g(self, eta_hat: np.ndarray, X_data: np.ndarray, 
                  N_data: np.ndarray, covariates: np.ndarray, 
                  alpha: float) -> np.ndarray:
        """Compute gradient of the log-likelihood with respect to eta parameters.
        
        Args:
            eta_hat: Current eta parameters
            X_data: Magnitude data
            N_data: Duration data
            covariates: Covariate matrix for eta
            alpha: Alpha parameter
            
        Returns:
            Gradient vector
        """
        if hasattr(covariates, 'values'):
            covariates = covariates.values
        
        if isinstance(alpha, (list, tuple, np.ndarray)):
            alpha = alpha[0]
            
        # Using gradient of negative log-likelihood
        exp_terms = np.exp(np.dot(covariates, eta_hat))
        common_denominator = 1 + alpha * X_data * exp_terms
        numerator = covariates * alpha * X_data[:, np.newaxis] * exp_terms[:, np.newaxis]
        
        return np.sum(
            (1 / alpha + N_data[:, np.newaxis]) * (numerator / common_denominator[:, np.newaxis]), 
            axis=0
        ) - np.sum(N_data[:, np.newaxis] * covariates, axis=0)

    def gradient_descent_g(self, eta_init: np.ndarray, X_data: np.ndarray, 
                          N_data: np.ndarray, alpha_val: float, 
                          covariates: np.ndarray, learning_rate: float, 
                          num_iterations: int, init_multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform gradient descent to estimate eta parameters.
        
        Args:
            eta_init: Initial eta parameters
            X_data: Magnitude data
            N_data: Duration data
            alpha_val: Alpha parameter
            covariates: Covariate matrix for eta
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of iterations
            init_multiplier: Multiplier for gradient updates
            
        Returns:
            Tuple containing:
                eta: Estimated eta parameters
                cost: Cost history
        """
        cost = np.array([])
        eta = eta_init.copy()
        for i in range(num_iterations):
            grad = self.gradient_g(eta, X_data, N_data, covariates, alpha_val)
            eta -= init_multiplier * learning_rate * grad
            cost = np.append(cost, self.mse(eta, eta_init))
        return eta, cost

    def gradient_h(self, beta_hat: np.ndarray, N_data: np.ndarray, 
                  covariates: np.ndarray) -> np.ndarray:
        """Compute gradient of the log-likelihood with respect to beta parameters.
        
        Args:
            beta_hat: Current beta parameters
            N_data: Duration data
            covariates: Covariate matrix for beta
            
        Returns:
            Gradient vector
        """
        if hasattr(covariates, 'values'):
            covariates = covariates.values
            
        # Uses gradient of negative log-likelihood
        exp_terms = np.exp(np.dot(covariates, beta_hat))
        return np.dot(
            N_data, (covariates * exp_terms[:, np.newaxis]) / (1 + exp_terms[:, np.newaxis])
        ) - np.sum(covariates, axis=0)

    def gradient_descent_h(self, beta_init: np.ndarray, N_data: np.ndarray, 
                          covariates: np.ndarray, learning_rate: float, 
                          num_iterations: int, init_multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform gradient descent to estimate beta parameters.
        
        Args:
            beta_init: Initial beta parameters
            N_data: Duration data
            covariates: Covariate matrix for beta
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of iterations
            init_multiplier: Multiplier for gradient updates
            
        Returns:
            Tuple containing:
                beta: Estimated beta parameters
                cost: Cost history
        """
        cost = np.array([])
        beta = beta_init.copy()
        for i in range(num_iterations):
            grad = self.gradient_h(beta, N_data, covariates)
            beta -= init_multiplier * learning_rate * grad
            cost = np.append(cost, self.mse(beta, beta_init))
        return beta, cost

    def fit(self, X_data: np.ndarray, N_data: np.ndarray, W_data: np.ndarray, 
           Z_data: np.ndarray, alpha_val: float, learning_rate: float = 0.00001, 
           num_iterations: int = 2000, init_multiplier: float = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit the GSMP GLM model to data.
        
        Args:
            X_data: Magnitude data
            N_data: Duration data
            W_data: Covariate matrix for eta
            Z_data: Covariate matrix for beta
            alpha_val: Alpha parameter
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of iterations
            init_multiplier: Multiplier for initial parameter values and gradient updates
            
        Returns:
            Tuple containing:
                eta_est: Estimated eta parameters
                beta_est: Estimated beta parameters
                eta_cost: Cost history for eta
                beta_cost: Cost history for beta
        """
        try:
            # Use exact matrix solution as initial values
            initial_eta = self.exact_matrix_solver_eta(W_data, X_data, N_data)
            initial_beta = self.exact_matrix_solver_beta(Z_data, N_data)
        except:
            # Fall back to random initialization if exact solution fails
            initial_eta = init_multiplier * norm.rvs(0, 1, size=np.shape(W_data)[1])
            initial_beta = init_multiplier * norm.rvs(0, 1, size=np.shape(Z_data)[1])
        
        # Estimate parameters using gradient descent
        self.eta_est, self.eta_cost = self.gradient_descent_g(
            initial_eta, X_data, N_data, alpha_val, W_data, learning_rate, num_iterations, init_multiplier
        )
        self.beta_est, self.beta_cost = self.gradient_descent_h(
            initial_beta, N_data, Z_data, learning_rate, num_iterations, init_multiplier
        )

        return self.eta_est, self.beta_est, self.eta_cost, self.beta_cost
    
    def predict(self, W_new: np.ndarray, Z_new: np.ndarray, alpha_val: float) -> Dict[str, np.ndarray]:
        """Make predictions for new data.
        
        Args:
            W_new: New covariate matrix for eta
            Z_new: New covariate matrix for beta
            alpha_val: Alpha parameter
            
        Returns:
            Dictionary containing:
                lambda: Predicted lambda values
                p: Predicted p values
                alpha: Alpha parameter value
        """
        if self.eta_est is None or self.beta_est is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        lambda_pred = np.exp(np.dot(W_new, self.eta_est))
        p_pred = np.exp(np.dot(Z_new, self.beta_est)) / (1 + np.exp(np.dot(Z_new, self.beta_est)))
        
        return {
            'lambda': lambda_pred,
            'p': p_pred,
            'alpha': alpha_val
        }