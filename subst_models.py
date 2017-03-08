import numpy as np


def init_subst(n_states):
    pi=np.random.dirichlet(np.repeat(1,n_states))
    er=np.random.dirichlet(np.repeat(1,n_states*(n_states-1)/2))
    return pi, er

def fnF81(pi):
    """Vectorize the function"""
    n_states = pi.shape[0]
    #state_freqs = np.array([pi[s] for s in states])
    Q = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                Q[i,j] = pi[states[j]]-1.0
            else:
                Q[i,j] = pi[states[j]]
    beta = 1/(1-np.dot(pi, pi))
    Q = Q*beta
    return Q, states

def fnJC(n_states):
    Q = np.ones((n_states, n_states))
    np.fill_diagonal(Q,np.repeat(1-n_states,n_states))
    print(Q)
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
    Pi = np.diag(pi)
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

