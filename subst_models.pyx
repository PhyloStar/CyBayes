import numpy as np
np.random.seed(1234)
from scipy.stats import dirichlet
import config

def ptF811(pi,d):
    """Compute the Probability matrix under a F81 model
    """
    n_states = pi.shape[0]
    beta = 1/(1-np.dot(pi, pi))
    x = np.exp(-beta*d)
    y = 1.0-x
    p_t = np.reshape(np.repeat(pi*y,n_states),(n_states,n_states)).T+np.eye(n_states)*x
    return p_t

def fnF81(pi):
    """Vectorize the function"""
    #n_states = pi.shape[0]
    Q = np.reshape(np.repeat(pi,config.N_CHARS),(config.N_CHARS,config.N_CHARS)).T-np.eye(config.N_CHARS)
    beta = 1/(1-np.dot(pi, pi))
    Q = Q*beta
    return Q

def fnJC():
    Q = np.ones((config.N_CHARS, config.N_CHARS))
    np.fill_diagonal(Q,np.repeat(1-config.N_CHARS,config.N_CHARS))
    #print(Q)
    beta = -1.0/np.dot(np.repeat(1.0/config.N_CHARS,config.N_CHARS),np.diag(Q))
    return Q*beta

