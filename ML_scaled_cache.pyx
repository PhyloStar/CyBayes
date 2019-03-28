import numpy as np
import config, sys
cimport numpy as np

cpdef matML_unscaled(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, float n_cats):
    cdef list LL_mats
    cdef dict LL_mat
    cdef int parent, child, scale_counter, i, parent_flag
    cdef dict p_t, rescale
    
    cdef list rescale_list = []
    cdef double [:] ll = np.zeros(n_sites)
    cdef double LL = 0.0
    LL_mats  = []

    for i, p_t in enumerate(tmats):
        LL_mat = {}
        scale_counter = 0
        rescale = {}
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
            
#            scale_counter += 1
            
#            if scale_counter % 50 == 0:
#                rescale[parent] = LL_mat[parent].max(axis=0)
#                print("In MatML ")
#                print("Before rescaling ", LL_mat[parent], rescale[parent])
#                if LL_mat[parent].min() < 0:
#                    print("Before rescaling ", i, parent, rescale[parent].max(), rescale[parent].min(), LL_mat[parent].max(),  LL_mat[parent].min())
#                LL_mat[parent] = LL_mat[parent]/rescale[parent]

#                print("After rescaling ", i, parent, rescale[parent].max(), rescale[parent].min(), LL_mat[parent].max(),  LL_mat[parent].min())
#                LL += np.sum(np.log(rescale[parent]))

        ll += np.dot(pi, LL_mat[root])/n_cats
#        LL_mats.append(LL_mat)
#        rescale_list.append(rescale)

    LL += np.sum(np.log(ll))

#    for rescale in rescale_list:
#        for parent in rescale:
#            LL += np.sum(np.log(rescale[parent]))

#    for k in rescale_list:
#        for p in rescale_list[k]:
#            LL += np.sum(np.log(rescale_list[k][p]))
    #print(LL_mats[-1])

    if np.isnan(LL):
        print("***********Likelihood is NAN*******")
        sys.exit(1)
        LL = -100000000000000000

    return LL

cpdef matML_scaled(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, int n_cats):
    cdef list LL_mats
    cdef dict LL_mat
    cdef int parent, child, scale_counter, i, parent_flag, scale_denom
    cdef dict p_t, rescale
    
    cdef list rescale_list = []

#    cdef np.ndarray[double, ndim=1] ll = np.zeros(n_sites)
    cdef np.ndarray[double, ndim=2] temp_ll = np.zeros((n_cats, n_sites))
    cdef np.ndarray[double, ndim=1] max_arr = np.zeros(n_sites)
    cdef np.ndarray[double, ndim=2] clipped_ll = np.zeros((n_cats, n_sites))

    cdef double LL = 0.0
    LL_mats  = []
    scale_denom = 10

    for i, p_t in enumerate(tmats):
        LL_mat = {}
        scale_counter = 0
        rescale = {}

        for parent, child in edges:
#            print("Transition probability", p_t[parent,child])
            if child <= n_taxa:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
#                print("LL at ",  parent,  LL_mat[parent], child, ll_mats[child])
            else:
                if parent not in LL_mat:
                    LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                else:
                    LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])
#                print("LL at ",  parent,  LL_mat[parent],  child, LL_mat[child])
            
            scale_counter += 1
            
            if scale_counter % scale_denom == 0:
                rescale[parent] = LL_mat[parent].max(axis=0)

                LL_mat[parent] = LL_mat[parent]/rescale[parent]

                temp_ll[i] += np.log(rescale[parent])

        temp_ll[i] += np.log(np.dot(pi, LL_mat[root])/n_cats)
        LL_mats.append(LL_mat)
        rescale_list.append(rescale)

    max_arr = temp_ll.max(axis=0)#Maximum Log value across rates. Dimension is number of sites
    clipped_ll = np.clip(temp_ll-max_arr, -10, 0)#Clip the exponential power to a minimum value. e^-10 = 0.000045.

    LL = np.sum(max_arr + np.log(np.sum(np.exp(clipped_ll), axis=0)))


#                if LL_mat[parent].min() < 0:
#                    print("Before rescaling ", i, parent, rescale[parent].max(), rescale[parent].min(), LL_mat[parent].max(),  LL_mat[parent].min())
#                print("After rescaling ", "rate =", i, "scale counter = ", scale_counter, "parent =", parent, rescale[parent].max(), rescale[parent].min(), LL_mat[parent].max(),  LL_mat[parent].min())
#        print("LL mat at root", LL_mat[root], "at rate ", i)
#        print("LL at root ", np.dot(pi, LL_mat[root])/n_cats, "at rate ",i)
#    print("Clipped LL", clipped_ll)
#    print("Max array across rate min ", max_arr.min(), "Maximum", max_arr.max(), "Lowest value of difference underflow ", (temp_ll-max_arr).min(), "Clipped min",clipped_ll.min())
#        ll += np.dot(pi, LL_mat[root]*rescale[root])/n_cats
#    LL += np.sum(np.log(ll))
#    print(LL_mats[-1])

    if np.isnan(LL):
        print("***********Likelihood is NAN*******")
        sys.exit(1)
        LL = -100000000000000000

    return LL, LL_mats, rescale_list

cpdef matML_scaled_cache(double[:] pi, int root, dict ll_mats, list edges, list tmats, int n_sites, int n_taxa, int n_cats, list cache_LL_Mats, list rescale_list, list nodes_recompute):
    cdef list LL_mats, new_rescale_list
    cdef dict LL_mat
    cdef int parent, child, scale_counter, i, scale_denom, parent_flag
    cdef dict p_t, rescale
    
    cdef double [:] ll = np.zeros(n_sites)
    cdef double LL = 0.0

    cdef np.ndarray[double, ndim=2] temp_ll = np.zeros((n_cats, n_sites))

    LL_mats, new_rescale_list  = [], [] #New cache LL Mats and Rescale list
    scale_denom = 10

    for i, p_t in enumerate(tmats):
        LL_mat = {}
        scale_counter = 0
        rescale = {}

        for parent, child in edges:
#            print("Parent = {}, child = {}".format(parent, child))
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
                LL_mat[parent] = cache_LL_Mats[i][parent]

            scale_counter += 1

            if scale_counter % scale_denom == 0: 
                if parent not in rescale_list[i]:
                    print("Parent to be scaled not in cache list ", parent, scale_counter)
                    sys.exit(1)
                rescale[parent] = rescale_list[i][parent]
                if parent in nodes_recompute:
                    rescale[parent] = LL_mat[parent].max(axis=0)
                    LL_mat[parent] = LL_mat[parent]/rescale[parent]
                temp_ll[i] += np.log(rescale[parent])

        temp_ll[i] += np.log(np.dot(pi, LL_mat[root])/n_cats)
        LL_mats.append(LL_mat)
        new_rescale_list.append(rescale)

    max_arr = temp_ll.max(axis=0)#Maximum Log value across rates. Dimension is number of sites
    clipped_ll = np.clip(temp_ll-max_arr, -10, 0)#Clip the exponential power to a minimum value. e^-10 = 0.000045.

    LL = np.sum(max_arr + np.log(np.sum(np.exp(clipped_ll), axis=0)))

    return LL, LL_mats, new_rescale_list


