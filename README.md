# Diffusion Models

Collection of diffusion models to simulate asset prices (stocks). 

Currently this only has a standard diffusion process using GBM and a jump diffusion process using a Poisson distribution.

## Setup

Install the **latest** stable version:

```
pip install git+git@github.com:kadereub/diffusion-models.git
```

## Example Usage:

```
import numpy as np
import diffusion as dif

# stock parameters
mu = np.array([0.05, 0.2]) # annual expected return (drift)
sigma = np.array([0.1, 0.2]) # annual std. deviation
corr = np.array([[1.0, 0.4],[0.4, 1.0]])
# simulate two assets over one year
res = dif.standard(s0=100, mu=mu, sigma=sigma, horizon=1.0, n_sims=1000, seed=42, corr=corr)
```
