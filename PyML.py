from collections import defaultdict
import numpy as np
import config
import itertools as it
import multiprocessing as mp

def cogMatMLworker(pi, edges, mrca, n_cats, ll_mats, p_t):
    LL_mat = defaultdict()
    for parent, child in edges:
        if child <= config.N_TAXA:
            if parent not in LL_mat:
                LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
#                        print(p_t[parent,child])
#                        print(ll_mats[child])
#                        print(LL_mat[parent])
            else:
                LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
        else:
            if parent not in LL_mat:
                LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
            else:
                LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])

    return np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))

#def test_cogMatMLwrkr(pi, p_t, llmat):
#    return np.sum(np.log(np.dot(pi, np.dot(p_t, ))))
    
def parCognateMatML(pi, root, ll_mats_list, edges, tmats, n_cats, mrca_list):
    p = mp.Pool(2)
    args_list = [(pi, edges, mrca, n_cats, ll_mat, p_t) for mrca, ll_mat in zip(mrca_list, ll_mats_list) for p_t in tmats]
    
    return sum(p.starmap(cogMatMLworker, args_list))
    

def cognateMatML2(pi, root, ll_mats_list, edges, tmats, n_cats, mrca_list):
    #weighs the likelihood of each meaning by a gamma site rate

    cache_LL_Mats = defaultdict()

    n_cog_sets = len(ll_mats_list)

    ll_mrca = np.zeros(n_cog_sets)

    LL = 0.0
    for k in range(n_cats):
        p_t = tmats[k]
        
        for i in range(n_cog_sets):
            ll_mats = ll_mats_list[i]
            mrca = mrca_list[i]

            LL_mat = defaultdict()
            mrca_flag = -1


            for parent, child in edges:
                if child <= config.N_TAXA:
                    if parent not in LL_mat:
                        LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
#                        print(p_t[parent,child])
#                        print(ll_mats[child])
#                        print(LL_mat[parent])
                    else:
                        LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
                else:
                    if parent not in LL_mat:
                        LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                    else:
                        LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])

                if parent == mrca: mrca_flag += 1

                if mrca_flag > 0 and parent == mrca:
                    mrca_ll =  np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
                    ll_mrca[i] += mrca_ll
                    cache_LL_Mats[k,i] = LL_mat
#                    break
    LL = np.sum(ll_mrca)

    return LL, cache_LL_Mats

def cacheCognateMatML2(pi, root, ll_mats_list, cache_LL_Mats, nodes_recompute, edges, tmats, n_cats, mrca_list):
    LL_mats = defaultdict()
#    cdef dict LL_mat, ll_mats
#    cdef int parent, i, child, mrca
#    cdef dict p_t

#    cdef int n_cog_sets = len(ll_mats_list)
    n_cog_sets = len(ll_mats_list)

#    cdef double [:] ll_mrca = np.zeros(n_cog_sets)
    ll_mrca = np.zeros(n_cog_sets)

    for k in range(n_cats):
        p_t = tmats[k]
        for i in range(n_cog_sets):
            ll_mats = ll_mats_list[i]
            mrca = mrca_list[i]

            LL_mat = defaultdict()
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

                if parent == mrca: mrca_flag += 1

                if mrca_flag > 0 and parent == mrca:
                    mrca_ll =  np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
                    ll_mrca[i] += mrca_ll
                    LL_mats[k,i] = LL_mat
#                    break
    LL = np.sum(ll_mrca)

    return LL, LL_mats
