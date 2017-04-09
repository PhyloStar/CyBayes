import numpy as np
from scipy.stats import dirichlet
import random, math
import config

dir_alpha = 100.0
scaler_alpha = 1.25
epsilon = 1e-10

def mvDirichlet(pi):
    pi_new = dirichlet.rvs(dir_alpha*pi)[0]
    #print(pi, pi_new)
    hastings_ratio = dirichlet.logpdf(pi, pi_new) - dirichlet.logpdf(pi_new, pi)
    return pi_new, hastings_ratio


def mvScaler(x):
    log_c = scaler_alpha*(np.random.uniform()-0.5)
    c = math.exp(log_c)
    x_new = x*c
    return x_new, log_c

def mvVecScaler(X):
    log_c = scaler_alpha*(np.random.uniform()-0.5)
    c = math.exp(log_c)
    X_new = X*c
    return X_new, log_c


def mvSlider(x, a, b):
    """ a and b are bounds
    """
    x_hat = np.random.uniform(x-0.5, x+0.5)
    if x_hat < a:
        return 2.0*a -x_hat
    elif x_hat > b:
        return 2.0*b -x_hat
    else:
        return x_hat 
