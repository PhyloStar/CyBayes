import numpy as np
import config
from cython.parallel import prange

cpdef matML1(dict state, list taxa, dict ll_mats):
    LL_mats = []
    LL_mat = {}
    cdef int root, parent, i
    #cdef double[:] p_t, pi
    cdef list edges
    #cdef dict p_t
    
    root = state["root"]
    p_ts = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]
    ll = np.zeros((config.N_CATS, config.N_SITES))
    ll = np.zeros(config.N_SITES)

    for i, p_t in enumerate(p_ts):
        LL_mat = {}
        for parent, child in edges:
            if child <= config.N_TAXA:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
            else:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])

        x = np.dot(pi, LL_mat[root])/config.N_CATS
        #print(x)
        ll += np.log(x)
        #ll[i] = x
        LL_mats.append(LL_mat)
    LL = np.sum(ll)
    return LL, LL_mats

cpdef matML(dict state, list taxa, dict ll_mats):
    LL_mats = []
    cdef dict LL_mat = {}
    cdef int root, parent, i, child
    #cdef double[:] p_t, pi
    cdef list edges, p_ts
    #cdef double[:] pi
    cdef dict p_t
    cdef int n_cats = config.N_CATS
    cdef float LL
    cdef int n_taxa = config.N_TAXA

    root = state["root"]
    p_ts = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]
    ll = np.zeros((n_cats,config.N_SITES))

    for i, p_t in enumerate(p_ts):
        LL_mat = {}
        for parent, child in edges:
            if child <= n_taxa:
                if parent not in LL_mat:
                    #print(p_ts[i])
                    LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
            else:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])
        x = np.dot(pi, LL_mat[root])/n_cats
        #print(x)
        ll[i] = x
    #print(ll)
    LL = np.sum(np.log(ll))
    return LL, LL_mats


cpdef cache_matML(dict state, list taxa, dict ll_mats, dict cache_LL_Mat, list nodes_recompute):
    cdef dict LL_mat = {}
    cdef int root, parent
    #cdef double[:] p_t, pi
    cdef list edges
    #cdef dict p_t
    
    root = state["root"]
    p_t = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]

    for parent, child in edges:
        
        if parent in nodes_recompute:
            if child <= config.N_TAXA:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
            else:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])
        else:
            LL_mat[parent] = cache_LL_Mat[parent]#.copy()
    #print(pi, LL_mat[root])
    #print(np.sum(np.log(np.dot(pi, LL_mat[root]))))
    #ll = np.sum(np.log(np.dot(pi, LL_mat[root])))
    #ll = np.sum(np.log(np.sum(np.dot(pi, LL_mat[root]),axis =0)))
    ll = np.sum(np.log(np.dot(pi, LL_mat[root])))
    return ll, LL_mat


