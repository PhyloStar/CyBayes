import numpy as np
import config
<<<<<<< HEAD
from cython.parallel import prange

cpdef matML(dict state, list taxa, dict ll_mats):
    cdef list LL_mats = []
    cdef dict LL_mat = {}
    cdef int root, parent
=======
cimport numpy as np
#from fast_utils import fast_dot_blas
#cpdef matML(dict state, list taxa, dict ll_mats):

cpdef matML(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, float n_cats):
    cdef list LL_mats
    cdef dict LL_mat
    cdef int parent, child
>>>>>>> dev
    #cdef double[:] p_t, pi
    #cdef list edges
    cdef dict p_t
    
<<<<<<< HEAD
    root = state["root"]
    p_ts = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]
    #ll = np.zeros((config.N_CATS, config.N_SITES))
    ll = np.zeros(config.N_SITES)

    for p_t in p_ts:
=======
    #root = state["root"]
    #p_ts = state["transitionMat"]
    #pi = state["pi"]
    #edges = state["postorder"]
    #ll = np.zeros(config.N_SITES)
    cdef double [:] ll = np.zeros(n_sites)
    LL_mats  = []
    for p_t in tmats:
>>>>>>> dev
        LL_mat = {}
        for parent, child in edges:
            if child <= n_taxa:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
            else:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                else:
<<<<<<< HEAD
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])

        ll += np.dot(pi, LL_mat[root])
=======
                    #X = p_t[parent,child].dot(LL_mat[child])
                    #LL_mat[parent] *= X
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])
                
        ll += np.dot(pi, LL_mat[root])/n_cats
>>>>>>> dev
        LL_mats.append(LL_mat)
    LL = np.sum(np.log(ll/config.N_CATS))
    #print(LL_mats[-1])
    return LL, LL_mats

<<<<<<< HEAD
cpdef cache_matML(dict state, list taxa, dict ll_mats, list cache_LL_Mats, list nodes_recompute):
    LL_mats = []
    cdef dict LL_mat = {}
    cdef int root, parent
=======
cpdef matML_cython(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, int n_cats):
    cdef list LL_mats
    cdef dict LL_mat
    cdef int parent, child, i, j, k
>>>>>>> dev
    #cdef double[:] p_t, pi
    #cdef list edges
    cdef dict p_t
    
    #root = state["root"]
    #p_ts = state["transitionMat"]
    #pi = state["pi"]
    #edges = state["postorder"]
    #ll = np.zeros(config.N_SITES)
    cdef double [:] ll = np.zeros(n_sites)
    LL_mats  = [0]*n_cats
    for i in range(n_cats):
        LL_mat = {}
        for j in range(2*n_taxa-2):
            parent, child = edges[j][0], edges[j][1]
            p_t = tmats[i]
            if child <= n_taxa:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
            else:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                else:
                    #X = p_t[parent,child].dot(LL_mat[child])
                    #LL_mat[parent] *= X
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])
        ll += np.dot(pi, LL_mat[root])/(n_cats*1.0)
        LL_mats[i] = LL_mat
    LL = np.sum(np.log(ll))
    return LL, LL_mats

#cpdef cache_matML(dict state, list taxa, dict ll_mats, list cache_LL_Mats, list nodes_recompute):

cpdef cache_matML(double[:] pi, int root, dict ll_mats, list cache_LL_Mats, list nodes_recompute, list edges, list tmats, int n_sites, int n_taxa, int n_cats):
    cdef list LL_mats = []
    cdef dict LL_mat
    #cdef int root, parent, i
    cdef int parent, i, child
    #cdef double[:] p_t, pi
    #cdef list edges
    cdef dict p_t
    
    #root = state["root"]
    #p_ts = state["transitionMat"]
    #pi = state["pi"]
    #edges = state["postorder"]
    #ll = np.zeros(config.N_SITES)
    cdef double [:] ll = np.zeros(n_sites)

    for i, p_t in enumerate(tmats):
        LL_mat = {}
        for parent, child in edges:
            if parent in nodes_recompute:
                if child <= n_taxa:
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
                LL_mat[parent] = cache_LL_Mats[i][parent]#.copy()
        ll += np.dot(pi, LL_mat[root])/(n_cats*1.0)
        LL_mats.append(LL_mat)
    LL = np.sum(np.log(ll))
    return LL, LL_mats


cpdef matML1(dict state, list taxa, dict ll_mats):
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
<<<<<<< HEAD
    ll = np.zeros(config.N_SITES)
=======
    ll = np.zeros((n_cats,config.N_SITES))
>>>>>>> dev

    for i, p_t in enumerate(p_ts):
        LL_mat = {}
        for parent, child in edges:
<<<<<<< HEAD
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
                LL_mat[parent] = cache_LL_Mats[i][parent]#.copy()
        ll += np.dot(pi, LL_mat[root])
        LL_mats.append(LL_mat)
    LL = np.sum(np.log(ll/config.N_CATS))
    return LL, LL_mats


cpdef matML1(dict state, list taxa, dict ll_mats):
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
=======
>>>>>>> dev
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
<<<<<<< HEAD




=======
>>>>>>> dev
