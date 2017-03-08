import numpy as np

def fnF81(pi):
    states = list(pi.keys())
    n_states = len(states)
    state_freqs = np.array([pi[s] for s in states])
    Q = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                Q[i,j] = pi[states[j]]-1.0
            else:
                Q[i,j] = pi[states[j]]
    beta = 1/(1-np.dot(state_freqs, state_freqs))
    Q = Q/beta
    return Q, states

def fnJC(n_states):
    Q = np.ones((n_states, n_states))
    np.fill_diagonal(Q,np.repeat(1-n_states,n_states))
    print(Q)
    beta = -1.0/np.dot(np.repeat(1.0/n_states,n_states),np.diag(Q))
    return Q*beta

def fnGTR(er, pi):
    pass
