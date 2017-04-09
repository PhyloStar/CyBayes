
import numpy as np
np.random.seed(1234)
from scipy.stats import dirichlet

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
    n_states = pi.shape[0]
    Q = np.reshape(np.repeat(pi,n_states),(n_states,n_states)).T-np.eye(n_states)
    beta = 1/(1-np.dot(pi, pi))
    Q = Q*beta
    return Q

def fnJC(n_states):
    Q = np.ones((n_states, n_states))
    np.fill_diagonal(Q,np.repeat(1-n_states,n_states))
    #print(Q)
    beta = -1.0/np.dot(np.repeat(1.0/n_states,n_states),np.diag(Q))
    return Q*beta

