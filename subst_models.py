import numpy as np
np.random.seed(1234)
from scipy.stats import dirichlet

def init_pi_er(n_states, model):
    if model == "JC":
        pi = np.repeat(1.0/n_states, n_states)
    elif model in ["F81", "GTR"]:
        pi=np.random.dirichlet(np.repeat(1,n_states))
    er=np.random.dirichlet(np.repeat(1,n_states*(n_states-1)/2))
    return pi, er

def ptF81(pi,d):
    """Compute the Probability matrix under a F81 model
    """
    n_states = pi.shape[0]
    beta = 1/(1-np.dot(pi, pi))
    x = np.exp(-beta*d)
    y = 1.0-x
    p_t = np.reshape(np.repeat(pi*y,n_states),(n_states,n_states)).T+np.eye(n_states)*x
    return p_t

def ptJC(n_states, d):
    """Compute the Probability matrix under a F81 model
    """
    #n_states = pi.shape[0]
    pi = 1.0/n_states
    beta = 1.0/(1.0-pi)
    x = np.exp(-beta*d)
    y = 1.0-x
    p_t = np.reshape(np.repeat(pi*y,n_states*n_states),(n_states,n_states)).T+np.eye(n_states)*x
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

def fnGTR(er, pi):
    n_states = pi.shape[0]
    n_rates = er.shape[0]
    R, Pi = np.zeros((n_states, n_states)), np.zeros((n_states, n_states))
    iu1 = np.triu_indices(n_states,1)
    #il1 = np.tril_indices(n_states,-1)
    R[iu1] = er
    R = R+R.T
    #R[il1] = er

    X = np.diag(-np.dot(pi,R)/pi)
    R = R + X
    Pi = np.eye(n_states)*pi
    #print("pi ", pi)
    #print("er ", er)
    #print("R ", R)

    Q = np.dot(R,Pi)
    Q += np.diag(-np.sum(Q,axis=-1))
    beta = -1.0/np.dot(pi,np.diag(Q))
    Q = Q*beta
    #print("\n")
    #print(np.dot(pi,Q))
    return Q

