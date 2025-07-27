"""
GSMP-VGLM: A library for GSMP-VGLM
"""

# Package version
__version__ = "0.1.0"

# Import and expose key components that users will need
from .data_generation import TETLG_GLMData, GSMP_GLMData
from .grad_desc_sem_functions import GSMP_GLM
from .bayes_alpha_opt import case_1_2_log_like, bayesian_optimization, plot_optimization_convergence
from .goodness_of_fit import GSMPGoodnessOfFit
