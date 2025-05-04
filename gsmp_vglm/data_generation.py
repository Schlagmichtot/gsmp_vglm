"""
Module for generating synthetic data from GSMP and GLM models.
"""

import numpy as np
from scipy.stats import uniform, norm, geom, expon, gamma
from typing import Dict, Tuple, Optional, Union, List


class BaseGenerator:
    """Base class for data generators."""
    
    def __init__(self):
        """Initialize the base generator."""
        pass

    def make_data(self):
        """Generate synthetic data.
        
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("make_data method must be implemented by subclasses")


class GLMData(BaseGenerator):
    """Generate data from a GLM model with specified parameters."""
    
    def __init__(
        self, 
        n_obs: int, 
        n_cov: int, 
        dist_type: str, 
        add_intercept: bool, 
        eta_vec: np.ndarray, 
        beta_vec: np.ndarray
    ):
        """Initialize the GLM data generator.
        
        Args:
            n_obs: Number of observations to generate
            n_cov: Number of covariates
            dist_type: Distribution type ('uniform', 'gaussian', or 'exponential')
            add_intercept: Whether to add an intercept term to covariates
            eta_vec: Vector of parameters for the lambda parameters
            beta_vec: Vector of parameters for the p parameters
        """
        super().__init__()
        self.n_obs = n_obs
        self.n_cov = n_cov
        self.dist_type = dist_type
        self.add_intercept = add_intercept
        self.eta_vec = eta_vec
        self.beta_vec = beta_vec

    def make_W(self) -> np.ndarray:
        """Generate the covariate matrix W for lambda parameters.
        
        Returns:
            W: Covariate matrix with shape (n_obs, n_cov) or (n_obs, n_cov+1) with intercept
        """
        if self.dist_type == 'uniform':
            W = uniform.rvs(0, 1, (self.n_obs, self.n_cov))
        elif self.dist_type == 'gaussian':
            W = norm.rvs(0, 1, (self.n_obs, self.n_cov))
        elif self.dist_type == 'exponential':
            W = expon.rvs(scale=1, size=(self.n_obs, self.n_cov))
        else:
            raise ValueError('dist must be either "uniform", "gaussian" or "exponential"')
    
        if self.add_intercept:
            W = np.append(np.ones([self.n_obs, 1]), W, axis=1)  
        return W
    
    def make_Z(self) -> np.ndarray:
        """Generate the covariate matrix Z for p parameters.
        
        Returns:
            Z: Covariate matrix with shape (n_obs, n_cov) or (n_obs, n_cov+1) with intercept
        """
        if self.dist_type == 'uniform':
            Z = uniform.rvs(0, 1, (self.n_obs, self.n_cov))
        elif self.dist_type == 'gaussian':
            Z = norm.rvs(0, 1, (self.n_obs, self.n_cov))
        elif self.dist_type == 'exponential':
            Z = expon.rvs(scale=1, size=(self.n_obs, self.n_cov))
        else:
            raise ValueError('dist must be either "uniform", "gaussian" or "exponential"')

        if self.add_intercept:
            Z = np.append(np.ones([self.n_obs, 1]), Z, axis=1)
        return Z

    def make_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data from the GLM model.
        
        Returns:
            Tuple containing:
                X: Vector of total magnitudes
                Y: Vector of maximums
                N: Vector of durations
                W: Covariate matrix for lambda
                Z: Covariate matrix for p
        """
        W = self.make_W()
        Z = self.make_Z()

        # Calculate lambda and p from covariates and parameters
        lambda_vec = np.exp(np.dot(W, self.eta_vec))
        p_vec = (np.exp(np.dot(Z, self.beta_vec))) / (1 + np.exp(np.dot(Z, self.beta_vec)))

        # Generate geometric durations
        N = geom.rvs(p_vec)
        
        # Placeholders for X and Y
        X = np.array([])
        Y = np.array([])
        
        # Dictionary to store individual observations for each event
        obs_data = {}

        # Generate data for each observation
        for i, n in enumerate(N):
            obs = expon.rvs(scale=1 / lambda_vec[i], size=n)
            obs_data[i] = obs
            X = np.append(X, np.sum(obs))
            Y = np.append(Y, np.max(obs))

            # Print progress for large datasets
            if i % 100 == 0 and i > 0:
                print(f'Finished generating {i} observations')
                
        return X, Y, N, W, Z


class GSMP_GLMData(BaseGenerator):
    def __init__(self, n_obs, n_cov, dist_type, add_intercept, alpha, eta_vec, beta_vec):
        super().__init__()
        self.n_obs = n_obs
        self.n_cov = n_cov
        self.dist_type = dist_type
        self.add_intercept = add_intercept
        self.alpha = alpha  # Alpha parameter for GSMP model
        self.eta_vec = eta_vec
        self.beta_vec = beta_vec
        
    def make_W(self):
        if self.dist_type == 'uniform':
            W = uniform.rvs(0, 1, (self.n_obs, self.n_cov))
        elif self.dist_type == 'gaussian':
            W = norm.rvs(0, 1, (self.n_obs, self.n_cov))
        elif self.dist_type == 'exponential':
            W = expon.rvs(scale = 1, size = (self.n_obs, self.n_cov))
        else:
            raise ValueError('dist must be either "uniform", "gaussian" or "exponential"')
    
        if self.add_intercept:
            W = np.append(np.ones([self.n_obs, 1]), W, axis = 1)  
        return W
    
    def make_Z(self):
        if self.dist_type == 'uniform':
            Z = uniform.rvs(0, 1, (self.n_obs, self.n_cov))
        elif self.dist_type == 'gaussian':
            Z = norm.rvs(0, 1, (self.n_obs, self.n_cov))
        elif self.dist_type == 'exponential':
            Z = expon.rvs(scale=1, size=(self.n_obs, self.n_cov))
        else:
            raise ValueError('dist must be either "uniform", "gaussian" or "exponential"')

        if self.add_intercept:
            Z = np.append(np.ones([self.n_obs, 1]), Z, axis=1)
        return Z
        
    def make_data(self):
        W = self.make_W()
        Z = self.make_Z()

        lambda_vec = np.exp(np.dot(W, self.eta_vec))
        p_vec = (np.exp(np.dot(Z, self.beta_vec))) / (1 + np.exp(np.dot(Z, self.beta_vec)))

        N = geom.rvs(p_vec)
        X = np.array([])
        Y = np.array([])
        obs_data = {}

        for i, n in enumerate(N):
            # Generate gamma mixing variable for GSMP model
            Z_gamma = gamma.rvs(1/self.alpha, scale=self.alpha, size=1)
            # Generate exponential observations
            E_vec = expon.rvs(scale=1/lambda_vec[i], size=n)
            # Apply mixing to get Pareto observations
            X_vec = E_vec / Z_gamma
            
            obs_data[i] = X_vec
            X = np.append(X, np.sum(X_vec))
            Y = np.append(Y, np.max(X_vec))
                
        return X, Y, N, W, Z