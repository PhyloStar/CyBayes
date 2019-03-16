#from collections import defaultdict
import numpy as np
import config
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_multiply(np.ndarray[double, ndim=2] u, np.ndarray[double, ndim=2] v):
    cdef int i, j, k
    cdef int m, n, p

    m = u.shape[0]
    n = u.shape[1]
    p = v.shape[1]

    cdef np.ndarray[double, ndim=2] res = np.zeros((m, p))

    with cython.nogil:
        for i in range(m):
            for j in range(p):
                res[i,j] = 0
                for k in range(n):
                    res[i,j] += u[i,k] * v[k,j]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def elemenwise_multiply(np.ndarray[double, ndim=2] u, np.ndarray[double, ndim=2] v):
    cdef int i, j, k
#    cdef int m, n, p

    m = u.shape[0]
    n = u.shape[1]

    cdef np.ndarray[double, ndim=2] res = np.zeros((m, n))

    with cython.nogil:
        for i in range(m):
            for j in range(n):
                res[i,j] = u[i,j]*v[i,j]
    return res


#cpdef cognateMatML2(double[:] pi, int root, list ll_mats_list, list edges, list tmats, int n_cats, list mrca_list):
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cognateMatML2(pi, root, ll_mats_list, edges, tmats, n_cats, mrca_list, n_cog_sets):
    #weighs the likelihood of each meaning by a gamma site rate
    cdef dict LL_mat
    cdef int parent, child
    cdef dict p_t
    cdef int i, mrca, mrca_flag, j, k
    cdef double LL 
    cdef double LL_root, mrca_ll
    cdef dict LL_mats = {}
#    cdef int n_cog_sets = len(ll_mats_list)

#    cdef double [:] ll_mrca = np.zeros(n_cog_sets)
    
    cdef dict ll_mats

    LL = 0.0
    LL_root = 0.0

    for k in range(n_cats):
        p_t = tmats[k]
        
        for i in range(n_cog_sets):
            ll_mats = ll_mats_list[i]
            mrca = mrca_list[i]

            LL_mat = {}
#            mrca_flag = -1

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

#                if parent == mrca:
#                    mrca_flag += 1

#                    if mrca_flag > 0:
#                        mrca_ll =  np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#                        ll_mrca[i] += mrca_ll
#                        cache_LL_Mats[k,i] = LL_mat
    #                    break

#            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL_root += np.sum(np.log(np.dot(pi, LL_mat[root])))
            LL_mats[k,i] = LL_mat
#    print(LL_root, LL, LL_root-LL)
#    LL = np.sum(ll_mrca)

    return LL, LL_mats



#cpdef cacheCognateMatML2(double[:] pi, int root, list ll_mats_list, dict cache_LL_Mats, list nodes_recompute, list edges, list tmats, int n_cats, list mrca_list):
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cacheCognateMatML2(pi, root, ll_mats_list, cache_LL_Mats, nodes_recompute, edges, tmats, n_cats, mrca_list, int n_cog_sets):
    cdef dict LL_mats = {}
    cdef dict LL_mat, ll_mats
    cdef int parent, i, child, mrca
    cdef dict p_t
    cdef double LL = 0.0
    cdef double LL_root = 0.0

#    cdef int n_cog_sets = len(ll_mats_list)

#    cdef double [:] ll_mrca = np.zeros(n_cog_sets)

    for k in range(n_cats):
        p_t = tmats[k]
        for i in range(n_cog_sets):
            ll_mats = ll_mats_list[i]
            mrca = mrca_list[i]

            LL_mat = {}
            mrca_flag = -1

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
                    LL_mat[parent] = cache_LL_Mats[k,i][parent]

#                if parent == mrca:
#                    mrca_flag += 1

#                    if mrca_flag > 0:
#                        mrca_ll =  np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#                        ll_mrca[i] += mrca_ll
#                        LL_mats[k,i] = LL_mat
#                        break

#            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL_root += np.sum(np.log(np.dot(pi, LL_mat[root])))
            LL_mats[k,i] = LL_mat
#    print(LL_root, LL, LL_root-LL)

#    LL = np.sum(ll_mrca)

    return LL, LL_mats

cpdef matML(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, int n_cats):
    """weighs the likelihood by Jensen's inequality. Requires mrca broadcasting and other stuff which will make it slow.
    Better to go for the original function with two for loops"""
    cdef list LL_mats = []
    cdef dict LL_mat
    cdef int parent, child
    cdef dict p_t
    cdef double LL = 0.0

    for p_t in tmats:
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
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])

        
        LL += np.sum(np.log(np.dot(pi, LL_mat[root])))
        LL_mats.append(LL_mat)
    return LL, LL_mats

cpdef cache_matML(double[:] pi, int root, dict ll_mats, list cache_LL_Mats, list nodes_recompute, list edges, list tmats, int n_sites, int n_taxa, int n_cats):
    cdef list LL_mats = []
    cdef dict LL_mat
    cdef int parent, i, child
    cdef dict p_t
    cdef double LL = 0.0
    cdef np.ndarray[double, ndim=1] ll = np.zeros(n_sites)

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

        LL += np.sum(np.log(np.dot(pi, LL_mat[root])))
        LL_mats.append(LL_mat)
    return LL, LL_mats


cpdef cognateMatML2_matrix_mult(pi, root, ll_mats_list, edges, tmats, n_cats, mrca_list):
    #weighs the likelihood of each meaning by a gamma site rate
    cdef dict LL_mat
    cdef int parent, child
    cdef dict p_t
    cdef int i, mrca, mrca_flag, j, k
    cdef double LL 
    cdef double LL_root, mrca_ll
    cdef dict LL_mats = {}
    cdef int n_cog_sets = len(ll_mats_list)

#    cdef double [:] ll_mrca = np.zeros(n_cog_sets)
    
    cdef dict ll_mats

    LL = 0.0

    for k in range(n_cats):
        p_t = tmats[k]
        
        for i in range(n_cog_sets):
            ll_mats = ll_mats_list[i]
            mrca = mrca_list[i]

            LL_mat = {}
#            mrca_flag = -1

            for parent, child in edges:
                if child <= config.N_TAXA:
                    if parent not in LL_mat:
#                        LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                        LL_mat[parent] = matrix_multiply(p_t[parent,child], ll_mats[child])
                    else:
#                        LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
                        LL_mat[parent] = elemenwise_multiply(LL_mat[parent], matrix_multiply(p_t[parent,child], ll_mats[child]))
                else:
                    if parent not in LL_mat:
#                        LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                        LL_mat[parent] = matrix_multiply(p_t[parent,child], LL_mat[child])
                    else:
                        LL_mat[parent] = elemenwise_multiply(LL_mat[parent], matrix_multiply(p_t[parent,child], LL_mat[child]))
#                        LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])

#                if parent == mrca:
#                    mrca_flag += 1

#                    if mrca_flag > 0:
#                        mrca_ll =  np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#                        ll_mrca[i] += mrca_ll
#                        cache_LL_Mats[k,i] = LL_mat
    #                    break

#            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL_mats[k,i] = LL_mat
#    LL = np.sum(ll_mrca)

    return LL, LL_mats


#cpdef cognateMatML2_swap(pi, root, ll_mats_list, edges, tmats, n_cats, mrca_list):
#    #weighs the likelihood of each meaning by a gamma site rate
#    cdef dict LL_mat
#    cdef int parent, child
#    cdef dict p_t
#    cdef int i, mrca, mrca_flag, j, k
#    cdef double LL 
#    cdef double LL_root, mrca_ll
#    cdef dict LL_mats = {}
#    cdef int n_cog_sets = len(ll_mats_list)

#    cdef double [:] ll_mrca = np.zeros(n_cog_sets)
#    
#    cdef dict ll_mats

#    LL = 0.0

#        
#    for i in range(n_cog_sets):
#        ll_mats = ll_mats_list[i]
#        mrca = mrca_list[i]

#        for k in range(n_cats):
#            p_t = tmats[k]

#            LL_mat = {}
#            mrca_flag = -1

#            for parent, child in edges:
#                if child <= config.N_TAXA:
#                    if parent not in LL_mat:
#                        LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
#                    else:
#                        LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
#                else:
#                    if parent not in LL_mat:
#                        LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
#                    else:
#                        LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])

##                if parent == mrca:
##                    mrca_flag += 1

##                    if mrca_flag > 0:
##                        mrca_ll =  np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
##                        ll_mrca[i] += mrca_ll
##                        cache_LL_Mats[k,i] = LL_mat
#    #                    break

##            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
#            LL_mats[k,i] = LL_mat
#    LL = np.sum(ll_mrca)

#    return LL, LL_mats

