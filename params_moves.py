import numpy as np
from scipy.stats import dirichlet
import random

dir_alpha = 100.0
scaler_alpha = 1.25

def mvDirichlet(pi):
    pi_new = np.random.dirichlet(dir_alpha*pi)
    hastings_ratio = dirichlet.logpdf(pi, pi_new) - dirichlet.logpdf(pi_new, pi)
    return pi_new, hastings_ratio

def mvDualSlider(pi):
    i, j = random.sample(range(pi.shape[0]),2 )
    sum_ij = pi[i]+pi[j]
    x = np.random.uniform(sum_ij)
    y = sum_ij -x
    pi[i], pi[j] = x, y
    return pi

def mvScaler(x):
    log_c = scaler_alpha*(np.random.uniform()-0.5)
    c = np.exp(log_c)
    x_new = x*c
    return x_new, log_c

def mvVecScaler(X):
    log_c = scaler_alpha*(np.random.uniform()-0.5)
    c = np.exp(log_c)
    X_new = X*c
    return X_new, log_c


def mvSlider(x, a, b):
    """ a and b are bounds
    """
    x_hat = np.random.uniform(x-0.5, x+0.5)
    if x_hat < a:
        return 2.0*a -x_hat
    elif xhat > b:
        return 2.0*b -x_hat
    else:
        return x_hat 
