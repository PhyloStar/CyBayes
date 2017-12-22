import numpy as np
import config


cpdef matML(dict state, list taxa, dict ll_mats):
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
        if child <= config.N_TAXA:
            if parent not in LL_mat:
                #print("Leaf ", child, " No parent ", parent, " before multiplication ", ll_mats[child])
                #print("prob. transition matrix ", p_t[parent,child])
                LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                #print("Leaf ", child, " parent ", parent, " after multiplication ", LL_mat[parent])
            else:
                #print("Leaf ", child, " parent ", parent, " before multiplication ", ll_mats[child])
                #print("prob. transition matrix ", p_t[parent,child])
                #print("Before multiplication ", LL_mat[parent])
                LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
                #print("Leaf ", child, " parent ", parent," after multiplication ", LL_mat[parent])
        else:
            if parent not in LL_mat:
                #print("Not leaf ", child, " No parent ", parent, " before multiplication ", LL_mat[child])
                #print("prob. transition matrix ", p_t[parent,child])            
                LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                #print("Not leaf ", child, " parent ", parent, "after multiplication ", LL_mat[parent])                
            else:
                #print("Before multiplication ", LL_mat[child])
                #print("prob. transition matrix ", p_t[parent,child])
                X = p_t[parent,child].dot(LL_mat[child])
                #print("New marginal ", X)
                #print("Before multiplication ", LL_mat[parent])
                LL_mat[parent] *= X
                #print("After multiplication ", LL_mat[parent])
    #print(pi, LL_mat[root])
    #print(np.sum(np.log(np.dot(pi, LL_mat[root]))))
    #ll = np.sum(np.log(np.sum(np.dot(pi, LL_mat[root]),axis =0)))
    #ll = np.sum(np.log(np.dot(pi, LL_mat[root])))
    ll = np.sum(np.log(np.dot(pi, LL_mat[root])))
    return ll, LL_mat

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


