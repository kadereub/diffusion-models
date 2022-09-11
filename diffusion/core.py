
from typing import Iterable
import numpy as np
from .utils import cholesky_numba


def gbm(n_assets, dt, corr, n_sims=1_000, seed=42):
    """
     Geometric Brownian motion.

    Args:
        n_assets (int): number of asset paths to simulate
        dt (float): size of the discrete time steps simulated
        corr (numpy.ndarray): If none assume uncorrelated asset paths else usses corr as the correlation matrix.
        n_sims (int, optional): number of Monte-Carlo simulations
        seed (int, optonal): the RNG seed.
    Return:
        numpy.ndarray: the simulated prices series
    """
    state_ = np.random.RandomState(seed)
    x = state_.normal(0, np.sqrt(dt), size=(n_assets, n_sims))
    l = cholesky_numba(corr)
    wt = np.dot(l, x)
    return wt


def standard(s0, mu, sigma, horizon, n_sims, seed=42, corr=None):
    """
    Standard diffusion process, using Geometric Brownian motion, for simulating asset prices.

    Args:
        horizon (float): period in years (for simulation)
        n_sims (int): number of Monte-Carlo simulations
        seed (int, optonal): the RNG seed
        corr (numpy.ndarray, optional): If none assume uncorrelated asset paths else usses corr as the correlation matrix
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
    
    dt = horizon / n_sims
    y = gbm(n_assets=n, dt=dt, corr=corr, n_sims=n_sims, seed=seed)
    s_t = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * y.T
    )
    s_t = np.vstack([np.ones(n), s_t])
    s_t = s0 * s_t.cumprod(axis=0)
    return s_t


