import numpy as np
from collections import defaultdict
from scipy import linalg
import subst_models as sm

n_states =4
p_02 = np.array([[ 0.825092,  0.084274,  0.045317,  0.045317],
       [ 0.084274,  0.825092,  0.045317,  0.045317],
       [ 0.045317,  0.045317,  0.825092,  0.084274],
       [ 0.045317,  0.045317,  0.084274,  0.825092]])


p_01 = np.array([[ 0.906563, 0.045855, 0.023791, 0.023791],
       [0.045855, 0.906563, 0.023791, 0.023791],
       [ 0.023791, 0.023791, 0.906563, 0.045855],
       [ 0.023791, 0.023791, 0.045855, 0.906563]])


Q = np.array([[-4, 1, 2, 1],[1, -4,1, 2],[2, 1, -4, 1], [1, 2, 1, -4]])/4.0

leaves = ["1","2","3","4","5"]
n_leaves = len(leaves)

edges_dict = {(7, "1"):0.2, (7, "2"):0.2, (6, 7):0.1, (6, "3"):0.2, (8, "4"):0.2, (8, "5"):0.2, (9, 6):0.1, (9, 8):0.1}
edges = [(7, "1"), (7, "2"), (6, 7), (6, "3"), (8, "4"), (8, "5"), (9, 6), (9, 8)]
edges_int = [(7, 1), (7, 2), (6, 7), (6, 3), (8, 4), (8, 5), (0, 6), (0, 8)]

pi = []
sites = {"1":1, "2":2, "3":3, "4":2,"5":2}
#site_mat = defaultdict()
#ll_mat = np.zeros(((2*n_leaves-1), n_states))

#print(ll_mat)

    #site_mat[k,v-1] = 1.0

#print(ll_mat)
#print(site_mat)


p_t = p_01


def matML():
    for parent, child in edges_int:
        if child <= n_leaves:
            p_t = p_02
        else:
            p_t = p_01
        if ll_mat[parent].all() == 0.0:
            ll_mat[parent] = np.dot(p_t,ll_mat[child])
        else:
            ll_mat[parent] = ll_mat[parent]*np.dot(p_t,ll_mat[child])
    return ll_mat


def ML(edges_dict):
    ll_mat = defaultdict(lambda: 1)
    zero_vec = np.zeros(n_states)
    for k, v in sites.items():
        x = np.zeros(n_states)
        x[v-1] = 1.0
        #site_mat[k] = x
        ll_mat[k] = x
    for parent, child in edges:
        p_t = linalg.expm(Q*edges_dict[parent,child])
        #print(parent, child,"\n")
        #print(p_t)
        #if int(child) <= n_leaves:
        #    p_t = p_02
        ll_mat[parent] = ll_mat[parent]*np.dot(p_t,ll_mat[child])
        #print(parent, child)
        #if int(child) <= n_leaves:
            #print(site_mat[child])
            #print(ll_mat[parent])
            #ll_mat[parent] = ll_mat[parent]*np.dot(p_02,site_mat[child])
            #if parent not in ll_mat:
            #    ll_mat[parent] = np.dot(p_02,site_mat[child])
            #else:
            #    ll_mat[parent] = ll_mat[parent]*np.dot(p_02,site_mat[child])
        #else:
            #ll_mat[parent] = ll_mat[parent]*np.dot(p_01,ll_mat[child])
            #if parent not in ll_mat:
            #    ll_mat[parent] = np.dot(p_01,ll_mat[child])
            #else:
            #    ll_mat[parent] = ll_mat[parent]*np.dot(p_01,ll_mat[child])
    
    return ll_mat

def state_ML(edges_dict, mcmc_state)

n_states = 4
#pi, er = sm.init_subst(n_states)
for i in range(10000):
    x = ML(edges_dict)
#ll_mat = matML()

print(x)

