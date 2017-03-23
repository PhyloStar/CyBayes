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

def ptF811(pi,d):
    """Compute the Probability matrix under a F81 model
    """
    n_states = pi.shape[0]
    beta = 1/(1-np.dot(pi, pi))
    x = np.exp(-beta*d)
    y = 1.0-x
    p_t = np.reshape(np.repeat(pi*y,n_states),(n_states,n_states)).T+np.eye(n_states)*x
    return p_t

def binaryptF81(pi, d):
    """Compute the probability matrix for binary characters
    """
    p_t = np.empty((2,2))
    beta = 1/(1-np.dot(pi, pi))
    x = np.exp(-beta*d)
    y = 1.0-x
    p_t[0][0] = pi[0]+pi[1]*x
    p_t[0][1] = pi[1]*y
    p_t[1][0] = pi[0]*y
    p_t[1][1] = pi[1]+pi[0]*x
    return p_t

def ptF81(pi,d):
    """Compute the Probability matrix under a F81 model
    """
    n_states = pi.shape[0]
    beta = 1/(1-np.dot(pi, pi))
    x = np.exp(-beta*d)
    y = 1.0-x
    p_t = np.array([pi*y]*n_states)+np.eye(n_states)*x
    #p_t = np.reshape(np.repeat(pi*y,n_states),(n_states,n_states)).T+np.eye(n_states)*x
    return p_t

def ptF811(pi,d):
    """Compute the Probability matrix under a F81 model
    """
    n_states = pi.shape[0]
    beta = 1/(1-np.dot(pi, pi))
    x = np.exp(-beta*d)
    y = 1.0-x
    #p_t = np.array([pi*y]*n_states)+np.eye(n_states)*x
    p_t = np.reshape(np.repeat(pi*y,n_states),(n_states,n_states)).T+np.eye(n_states)*x
    return p_t

def ptJC(n_states, d):
    """Compute the Probability matrix under a F81 model
    """
    #n_states = pi.shape[0]
    pi = 1.0/n_states
    beta = 1.0/(1.0-pi)
    x = np.exp(-beta*d)
    y = pi*(1.0-x)
    #p_t = np.full((n_states, n_states), y)
    #p_t = p_t + np.eye(n_states)*x
    p_t = np.empty((n_states, n_states))
    p_t.fill(y)
    np.fill_diagonal(p_t, x+y)
    #p_t = np.array([[y]*n_states]*n_states)+np.eye(n_states)*x
    #p_t = np.reshape(np.repeat(pi*y,n_states*n_states),(n_states,n_states)).T+np.eye(n_states)*x
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

def fnGTR1(er, pi):
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

def fnGTR(er, pi):
    n_states = pi.shape[0]
    n_rates = er.shape[0]
    R = np.zeros((n_states, n_states))
    iu1 = np.triu_indices(n_states,1)
    #il1 = np.tril_indices(n_states,-1)
    R[iu1] = er
    R = R+R.T
    #R[il1] = er

    Pi = np.diag(pi)
    Q = np.dot(R,Pi)
    
    #X = np.diag(-np.dot(pi,R)/pi)
    #R = R + X

    Q += np.diag(-np.sum(Q,axis=-1))
    beta = -1.0/np.dot(pi,np.diag(Q))
    Q = Q*beta
    #print("\n")
    #print(np.dot(pi,Q))
    return Q



