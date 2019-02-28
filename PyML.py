from collections import defaultdict
import numpy as np
#from config import N_TAXA
#import itertools as it

def cognateMatML2(pi, root, ll_mats_list, edges, tmats, n_cats, mrca_list, n_cog_sets):
    #weighs the likelihood of each meaning by a gamma site rate

    cache_LL_Mats = defaultdict()

    n_taxa = int((root+1)/2)

#    cache_LL_Mats = {}

    LL = 0.0

    for k in range(n_cats):      
        p_t = tmats[k]
        for i in range(n_cog_sets):
            ll_mats = ll_mats_list[i]
            mrca = mrca_list[i]

#            LL_mat = defaultdict()
            LL_mat = {}

            for parent, child in edges:
#                if child <= config.N_TAXA:
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

            LL += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            cache_LL_Mats[k,i] = LL_mat


    return LL, cache_LL_Mats

def cacheCognateMatML2(pi, root, ll_mats_list, cache_LL_Mats, nodes_recompute, edges, tmats, n_cats, mrca_list, n_cog_sets):
    LL_mats = defaultdict()
#    LL_mats = {}
    LL = 0.0

    n_taxa = int((root+1)/2)

    for k in range(n_cats):      
        p_t = tmats[k]
        for i in range(n_cog_sets):
            ll_mats = ll_mats_list[i]
            mrca = mrca_list[i]

            LL_mat = defaultdict()
#            LL_mat = {}

            for parent, child in edges:
                if parent in nodes_recompute:
#                    if child <= config.N_TAXA:
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
                    LL_mat[parent] = cache_LL_Mats[k,i][parent]

            LL += np.sum(np.log(np.dot(pi, LL_mat[mrca])))
            LL_mats[k,i] = LL_mat

    return LL, LL_mats



