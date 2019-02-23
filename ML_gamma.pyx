#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
import config
cimport numpy as np
#from fast_utils import fast_dot_blas
#cpdef matML(dict state, list taxa, dict ll_mats):

cpdef matML(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, float n_cats):
    cdef list LL_mats
    cdef dict LL_mat
    cdef int parent, child
    #cdef double[:] p_t, pi
    #cdef list edges
    cdef dict p_t
    
    #root = state["root"]
    #p_ts = state["transitionMat"]
    #pi = state["pi"]
    #edges = state["postorder"]
    #ll = np.zeros(config.N_SITES)
    cdef double [:] ll = np.zeros(n_sites)
    LL_mats  = []
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
                    #X = p_t[parent,child].dot(LL_mat[child])
                    #LL_mat[parent] *= X
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])

            print(parent, LL_mat[parent])
        
        print("PI shape ", pi.shape, LL_mat[root].shape)
        
        ll += np.dot(pi, LL_mat[root])/n_cats
        print("Log Likelihood ", np.asarray(ll).shape)
        LL_mats.append(LL_mat)
    LL = np.sum(np.log(ll))
    #print(LL_mats[-1])
    return LL, LL_mats

cpdef cognateMatML1(double[:] pi, int root, list ll_mats_list, list edges, list tmats, float n_cats, list mrca_list):
    # Weighs the likelihood of each alignment using Gamma site rates.
#    cdef list LL_mats
    cdef dict LL_mat
    cdef int parent, child
    cdef dict p_t
    cdef int i, mrca
    cdef float LL 
    cdef float LL_root

    cdef double [:] ll
#    LL_mats  = []

    i, LL, LL_root = 0, 0.0, 0.0

    for ll_mats in ll_mats_list:
#        cogs_taxa_list = cogset_taxa_list[i]
#        n_leaves = len(cogs_taxa_list)

        ll = np.zeros(ll_mats[1].shape[1])
        ll_root = np.zeros(ll_mats[1].shape[1])
        mrca = mrca_list[i]

        for p_t in tmats:
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

                if parent == mrca: break #break out of loop after computing the likelihood at the mrca
            
            ll += np.dot(pi, LL_mat[mrca])/n_cats
#            ll_root += np.dot(pi, LL_mat[root])/n_cats
            
        i += 1
        temp_ll = np.sum(np.log(ll))
        LL += temp_ll

    return LL

cpdef cognateMatML2(double[:] pi, int root, list ll_mats_list, list edges, list tmats, float n_cats, list mrca_list):
    #weighs the likelihood of each meaning by a gamma site rate
    cdef dict LL_mat
    cdef int parent, child
    cdef dict p_t
    cdef int i, mrca, mrca_flag
    cdef float LL 
    cdef float LL_root

    cdef double [:] ll_root_mrca = np.zeros(len(ll_mats_list))
    cdef double [:] ll_root = np.zeros(len(ll_mats_list))
    cdef double [:] ll_mrca = np.zeros(len(ll_mats_list))

    LL = 0.0
    for p_t in tmats:
        i, LL_root = 0, 0.0
        
        for i in range(len(ll_mats_list)):
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

                if parent == mrca: mrca_flag += 1

                if mrca_flag > 0 and parent == mrca:
                    mrca_ll =  np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#                    print("parent = {}, mrca_ll = {}".format(parent, mrca_ll))
                    ll_mrca[i] += mrca_ll
                    break


#            temp_root = np.sum(np.log(np.dot(pi, LL_mat[mrca]))) #mrca likelihood codeblock

#            print("cognate set = {}, mrca = {}, root = {}".format(i, mrca, root))
#            ll_root_mrca[i] += np.sum(np.log(np.dot(pi, LL_mat[mrca])/n_cats))
#            ll_root[i] += np.sum(np.log(np.dot(pi, LL_mat[root])/n_cats))

            i += 1
#    print("LL_mrca = {}, LL_root_mrca = {}, LL_root = {}".format(np.asarray(ll_mrca), np.asarray(ll_root_mrca), np.asarray(ll_root)))
#    print("LL_mrca = {}, LL_root_mrca = {}, LL_root = {}".format(np.sum(ll_mrca), np.sum(ll_root_mrca), np.sum(ll_root)))

#    LL = np.sum(ll_root)
    LL = np.sum(ll_mrca)

    return LL

cpdef cacheCognateMatML2(double[:] pi, int root, dict ll_mats, list cache_LL_Mats, list nodes_recompute, list edges, list tmats, int n_sites, int n_taxa, int n_cats):
    cdef list LL_mats = []
    cdef dict LL_mat
    cdef int parent, i, child

    cdef dict p_t
    
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

cpdef cognateMatML3(double[:] pi, int root, list ll_mats_list, list edges, list tmats, float n_cats, list mrca_list):
    #weighs the likelihood of each meaning without gamma site rate. Removing weighing also. Possible bug in the cognateMatML2 function
    cdef dict LL_mat
    
    cdef int parent, child, prev_parent
    cdef dict p_t
    cdef int i, mrca, j, n_branches
    cdef float LL, temp_root, temp_mrca

    cdef list LL_mats = []

    n_branches = len(edges)
    
    p_t = tmats[0]

    LL = 0.0
    i = 0

    for i in range(len(ll_mats_list)):

        ll_mats = ll_mats_list[i]
        mrca = mrca_list[i]

        LL_mat = {}
        prev_parent = edges[0][0]

        for j in range(n_branches):
            parent, child = edges[j][0], edges[j][1]

#            print("mrca = {}, parent = {}, child = {} ".format(mrca, parent, child))


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

#            prev_parent = parent
#            if prev_parent == mrca: break
#        temp_root = np.sum(np.log(np.dot(pi, LL_mat[mrca]))) #mrca likelihood

#            print("previous_parent = {}, parent = {}".format(prev_parent, parent))

        temp_root = np.sum(np.log(np.dot(pi, LL_mat[root])))

        LL += temp_root

    return LL

cpdef fastcognateMatML3(double[:] pi, int root, list ll_mats_list, list edges, list tmats, float n_cats, list mrca_list):
    #weighs the likelihood of each meaning without gamma site rate. Removing weighing also.
    # Does a fast calculation by initializing the LL_mats elsewhere.
    # Tested not so fast. Takes 7 minutes as comapred to cognateMatML3 that takes 5 minutes.

    cdef dict LL_mat
    
    cdef int parent, child, prev_parent
    cdef dict p_t
    cdef int i, mrca, j, n_branches
    cdef float LL, temp_root, temp_mrca

    cdef list LL_mats = []

    n_branches = len(edges)
    
    p_t = tmats[0]

    LL = 0.0
    i = 0

    for i in range(len(ll_mats_list)):

        ll_mats = ll_mats_list[i]
        mrca = mrca_list[i]

        LL_mat = {}
        prev_parent = edges[0][0]
        
        rows, cols = ll_mats[1].shape

        for j in range(n_branches):
            parent, child = edges[j][0], edges[j][1]
            LL_mat[parent] = np.ones((rows, cols))
            if child <= config.N_TAXA:
                LL_mat[child] = ll_mats[child]

        for j in range(n_branches):
            parent, child = edges[j][0], edges[j][1]
            LL_mat[parent] = LL_mat[parent]*p_t[parent,child].dot(LL_mat[child])

        temp_root = np.sum(np.log(np.dot(pi, LL_mat[root])))
        LL += temp_root

    return LL



cpdef cognateMatML_orig(double[:] pi, int root, list ll_mats_list, list edges, list tmats, list cogset_taxa_list, float n_cats, list mrca_list):
#    cdef list LL_mats
    cdef dict LL_mat
    cdef int parent, child
    cdef dict p_t
    cdef int i, mrca
    cdef float LL 
    cdef float LL_root

    cdef double [:] ll
#    LL_mats  = []

    i, LL, LL_root = 0, 0.0, 0.0

    for ll_mats in ll_mats_list:
        cogs_taxa_list = cogset_taxa_list[i]
        n_leaves = len(cogs_taxa_list)

        ll = np.zeros(ll_mats[1].shape[1])
        ll_root = np.zeros(ll_mats[1].shape[1])
        mrca = mrca_list[i]

        for p_t in tmats:
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

                if parent == mrca: break #break out of loop after computing the likelihood at the mrca
            
            ll += np.dot(pi, LL_mat[mrca])/n_cats
#            ll_root += np.dot(pi, LL_mat[root])/n_cats
            
        i += 1
#        temp_ll, temp_ll_root = np.sum(np.log(ll)), np.sum(np.log(ll_root))
        temp_ll = np.sum(np.log(ll))
        LL += temp_ll
#        LL_root += temp_ll_root
#        if temp_ll != temp_ll_root :
#            print("{}th cognate set, cognate set size = {}, mrca = {}, Likelihood parent={}, root={}, Total Likelihood Parent={}, Total Likelihood Root={}".format(i, n_leaves, mrca, temp_ll, temp_ll_root, LL, LL_root))

    return LL


cpdef matML_cython(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, int n_cats):
    cdef list LL_mats
    cdef dict LL_mat
    cdef int parent, child, i, j, k
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





#            print("parent = {}, prev_parent = {}".format(parent, prev_parent)) 
#            if prev_parent == mrca: break

#            print("parent = {}, child = {}, likelihood matrix = {}".format(parent, child, LL_mat[parent]))
#            print(parent, child)
#            print(LL_mat[parent])
#            prev_parent = parent
#            if parent == mrca: break #break out of loop after computing the likelihood at the mrca
#        if mrca != root:
#            print(root, mrca, "\n", LL_mat[root]-LL_mat[mrca])
#            print("root = {}, mrca = {}, root LL Vector = {}, mrca LL Vector = {}".format(root, mrca, np.log(np.dot(pi, LL_mat[root])), np.log(np.dot(pi, LL_mat[mrca]))))
#            temp_root = np.sum(np.log(np.dot(pi, LL_mat[root])))
#            temp_mrca = np.sum(np.log(np.dot(pi, LL_mat[mrca])))
#            print("root = {}, mrca = {}, root_LL = {}, mrca_LL = {}".format(root, mrca, temp_root, temp_mrca))
#            LL += temp_root

#        if mrca != root:
#            print("{}th cognate set, root = {}, mrca = {}, LL Difference Vector = {}".format(i, root, mrca, np.log(np.dot(pi, LL_mat[root]))-np.log(np.dot(pi, LL_mat[mrca]))))
