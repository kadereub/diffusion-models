
from typing import Iterable
import numpy as np
from .utils import cholesky_numba


def gbm(s0, mu, sigma, horizon, n_sims, seed=42, corr=None):
    """
    Standard diffusion process, using Geometric Brownian motion.

    Args:
        horizon (float): period in years (for simulation)
        n_sims (int): number of Monte-Carlo simulations
        seed (int, optonal): the RNG seed.
        corr (numpy.ndarray, optional): If none assume uncorrelated asset paths else usses corr as the correlation matrix.
    Return:
        numpy.ndarray: the simulated prices series
    """

    if isinstance(mu,Iterable) and isinstance(mu,Iterable):
        assert len(mu) == len(sigma), f"mu's shape {len(mu)} is not the same as sigma's shape {len(sigma)}"
        n = mu.shape[0]
    elif not isinstance(mu,Iterable) and not isinstance(mu,Iterable):
        n = 1
    else:
        raise ValueError(f"mu's shape {len(mu)} is not the same as sigma's shape {len(sigma)}")
    
    if corr is None:
        corr = np.zeros((n, n))
        np.fill_diagonal(corr, 1.0)
    
    state_ = np.random.RandomState(seed)
    dt = horizon / n_sims
    x = state_.normal(0, np.sqrt(dt), size=(n, n_sims))
    l = cholesky_numba(corr)
    y = np.dot(l, x)
    s_t = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * y.T
    )
    s_t = np.vstack([np.ones(n), s_t])
    s_t = s0 * s_t.cumprod(axis=0)
    return s_t


