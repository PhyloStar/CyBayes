import numpy as np

cpdef matML(state, list taxa, dict ll_mats):
    cdef dict LL_mat = {}
    cdef int root, parent
    #cdef double[:] p_t, pi
    cdef list edges
    
    root = state["root"]
    p_t = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]

    for parent, child in edges[::-1]:
        if child in taxa:
            if parent not in LL_mat:
                LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
            else:
                LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
        else:
            if parent not in LL_mat:
                LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
            else:
                LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])
    ll = np.sum(np.log(np.dot(pi, LL_mat[root])))
    return ll, LL_mat

cpdef cache_matML(state, list taxa, dict ll_mats, dict cache_LL_Mat, list nodes_recompute):
    cdef dict LL_mat = {}
    cdef int root, parent
    #cdef double[:] p_t, pi
    cdef list edges
    
    root = state["root"]
    p_t = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]

    for parent, child in edges[::-1]:
        
        if parent in nodes_recompute:
            if child in taxa:
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
        
    ll = np.sum(np.log(np.dot(pi, LL_mat[root])))
    return ll, LL_mat


