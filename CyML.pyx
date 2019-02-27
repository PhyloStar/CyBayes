#from collections import defaultdict
import numpy as np
import config
cimport numpy as np

#cpdef cognateMatML2(double[:] pi, int root, list ll_mats_list, list edges, list tmats, int n_cats, list mrca_list):
cpdef cognateMatML2(pi, root, ll_mats_list, edges, tmats, n_cats, mrca_list):
    #weighs the likelihood of each meaning by a gamma site rate
    cdef dict LL_mat
    cdef int parent, child
    cdef dict p_t
    cdef int i, mrca, mrca_flag, j, k
    cdef double LL 
    cdef double LL_root, mrca_ll
    cdef dict LL_mats = {}
    cdef int n_cog_sets = len(ll_mats_list)

    cdef double [:] ll_mrca = np.zeros(n_cog_sets)
    
    cdef dict ll_mats

    LL = 0.0
    for k in range(n_cats):
        p_t = tmats[k]
        
        for i in range(n_cog_sets):
            ll_mats = ll_mats_list[i]
            mrca = mrca_list[i]

            LL_mat = {}
            mrca_flag = -1

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
            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL_mats[k,i] = LL_mat
    LL = np.sum(ll_mrca)

    return LL, LL_mats


#cpdef cacheCognateMatML2(double[:] pi, int root, list ll_mats_list, dict cache_LL_Mats, list nodes_recompute, list edges, list tmats, int n_cats, list mrca_list):
cpdef cacheCognateMatML2(pi, root, ll_mats_list, cache_LL_Mats, nodes_recompute, edges, tmats, n_cats, mrca_list):
    cdef dict LL_mats = {}
    cdef dict LL_mat, ll_mats
    cdef int parent, i, child, mrca
    cdef dict p_t

    cdef int n_cog_sets = len(ll_mats_list)

    cdef double [:] ll_mrca = np.zeros(n_cog_sets)

    for k in range(n_cats):
        p_t = tmats[k]
        for i in range(config.N_COG_CLASSES):
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
            ll_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL_mats[k,i] = LL_mat

    LL = np.sum(ll_mrca)

    return LL, LL_mats

cpdef matML(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, int n_cats):
    #weighs the likelihood by Jensen's inequality
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

        LL += np.sum(np.log(np.dot(pi, LL_mat[root])))
        LL_mats.append(LL_mat)
    return LL, LL_mats



