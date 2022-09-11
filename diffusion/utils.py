
import numpy as np
import numba as nb
from scipy import linalg


def cov_mat(correlations, n_assets):
    """
    Creates a covariance matrix from a vector of correlations.

    Args:
        correlations (numpy.ndarray): The vector of correlations should be size (n^2 + n) / 2  - n.
        n_assets (int): The number of assets i.e. the matrix will be n x n.
    Return:
        numpy.ndarray: n x n covariance matrix of the assets.
    """

    # Create coveraince matrix
    n = n_assets
    cov = np.zeros((n, n))
    n_var = int(n * (n + 1) // 2 - n)
    # Upper & Lower triangular mask
    triu_mask = np.triu(np.ones(cov.shape, 'bool'), k=1)
    cov[triu_mask] = correlations
    cov = np.triu(cov)
    cov = cov + cov.T - np.diag(np.diag(cov))
    np.fill_diagonal(cov, 1.0)
    # TODO: Switch to using the numba implementation
    cov = linalg.cholesky(cov ** 2, lower=True) # using covariance matrix
    return cov


@nb.njit('float64[:, :](float64[:, :])')
def cholesky_numba(A):
    """
    A cholesky decomposition in Numba.
    [1] https://stackoverflow.com/questions/70133315/how-to-implement-cholesky-decomposition-using-numpy-efficiently
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1):
            s = 0
            for k in range(j):
                s += L[i][k] * L[j][k]

            if (i == j):
                L[i][j] = (A[i][i] - s) ** 0.5
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L


def cov_to_correl(covariance):
    """ Converts a covariance matrix to a correlation matrix"""
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation