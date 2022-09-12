
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
        numpy.ndarray: n_assets x n_sims array of the simulated prices series
    
    Examples:
    Simple structure to show dimensions
    >>> corr = np.array([[1.0, -0.2, 0.5], [-0.2, 1.0, -0.8], [0.5, -0.8, 1.0]])
    >>> res = gbm(3, 1/252, corr)
    >>> res.shape
    (3, 1000)

    Error with incorrect correlation matrix dimentions
    >>> corr = np.array([[1.0, -0.2, 0.5], [-0.2, 1.0, -0.8], [0.5, -0.8, 1.0]])
    >>> res = gbm(4, 1/252, corr)
    Traceback (most recent call last):
    ...
    AssertionError: correlation matrix dimentions should match number of assets
    """
    assert n_assets == corr.shape[0], f"correlation matrix dimentions should match number of assets"
    state_ = np.random.RandomState(seed)
    x = state_.normal(0, np.sqrt(dt), size=(n_assets, n_sims))
    l = cholesky_numba(corr)
    wt = np.dot(l, x)
    return wt


def poisson_jump(n_assets, dt, lm, mj, sj, n_sims=1_000, seed=42):
    """
     A poission distributed jump intensity.

    Args:
        n_assets (int): number of asset paths to simulate
        dt (float): size of the discrete time steps simulated
        lm (float/numpy.1darray): The lambda of the Poisson process i.e. intensity of jump e.g. number of jumps per annum
        mj (float/numpy.1darray): the meean of jump size
        sj (float/numpy.1darray): standard deviation of jump
        n_sims (int, optional): number of Monte-Carlo simulations
        seed (int, optonal): the RNG seed.
    Return:
        numpy.ndarray: n_assets x n_sims array of the simulated price jumps
    
    """
    state_ = np.random.RandomState(seed)
    p = state_.poisson(lm * dt, size=(n_assets, n_sims))
    j = state_.normal(mj, sj, size=(n_assets, n_sims))
    jumps = np.cumsum(p * j, axis=1)
    return jumps


def standard(s0, mu, sigma, horizon, n_sims, seed=42, corr=None):
    """
    Standard diffusion process, using Geometric Brownian motion, for simulating asset prices.

    Args:
        s0 (float/numpy.1darray): The initial stock prices
        mu (float/numpy.1darray): The annual drift (expected return)
        sigma (float/numpy.1darray): The annual std. dev. (volatility)
        horizon (float): period in years (for simulation)
        n_sims (int): number of Monte-Carlo simulations
        seed (int, optonal): the RNG seed
        corr (numpy.ndarray, optional): If none assume uncorrelated asset paths else usses corr as the correlation matrix
    Return:
        numpy.ndarray: the simulated prices series

    Examples:
    Simple run for two assets
    >>> res = standard(s0=100, mu=np.array([0.05, 0.2]), sigma=np.array([0.1, 0.2]), horizon=1.0, n_sims=1000, seed=42)
    >>> res.shape
    (1001, 2)

    Run with two correlated assets
    >>> res = standard(s0=100, mu=np.array([0.05, 0.2]), sigma=np.array([0.1, 0.2]), horizon=1.0, n_sims=1000, seed=42, corr=np.array([[1.0, 0.4],[0.4, 1.0]]))
    >>> np.round(res.mean(), 3)
    122.407
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


def jump(s0, mu, sigma, horizon, n_sims, seed=42, corr=None, **poi_kwargs):
    """
    Merton's jump diffusion process, using Geometric Brownian motion, for simulating asset prices.

    Args:
        s0 (float/numpy.1darray): The initial stock prices
        mu (float/numpy.1darray): The annual drift (expected return)
        sigma (float/numpy.1darray): The annual std. dev. (volatility)
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

    # TODO: Test this implementation
    jumps = poisson_jump(n_assets=n, dt=dt, lm=lm, mj=mj, sj=sj, n_sims=n_sims, seed=seed)

    s_t = np.exp(
        (mu - sigma ** 2 / 2 - lm * (mj  + sj ** 2 / 2)) * dt
        + sigma * y.T
        + jumps
    )

    s_t = np.vstack([np.ones(n), s_t])
    s_t = s0 * s_t.cumprod(axis=0)
    return s_t


if __name__ == "__main__":
    import doctest
    doctest.testmod()