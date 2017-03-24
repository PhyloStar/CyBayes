from collections import defaultdict
import numpy as np

def matML_dot(state, taxa, ll_mats):
    LL_mat = defaultdict()
    root = state["root"]
    p_t = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]
    
    for parent, child in edges[::-1]:
        if child in taxa:
            if parent not in LL_mat:
                #print(child, ll_mats[child], "\n", ll_mats[child].shape,"\n")
                #print(p_t[parent,child])
                LL_mat[parent] = np.dot(p_t[parent,child], ll_mats[child])
                
            else:
                LL_mat[parent] *= np.dot(p_t[parent,child], ll_mats[child])
        else:
            if parent not in LL_mat:
                LL_mat[parent] = np.dot(p_t[parent,child], LL_mat[child])
            else:
                LL_mat[parent] *= np.dot(p_t[parent,child], LL_mat[child])
        #print(parent, LL_mat[parent])
    #print("Root ", root,LL_mat[root], LL_mat[root].shape,pi, pi.shape, sep="\n")
    #print(np.dot(pi, LL_mat[root]))
    ll = np.sum(np.log(np.dot(pi, LL_mat[root])))
    return ll

def matML_inplace(state, taxa, ll_mats):
    LL_mat = defaultdict()
    root = state["root"]
    p_t = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]
    

    for parent, child in edges[::-1]:
        if child in taxa:
            if parent not in LL_mat:
                #print(child, ll_mats[child], "\n", ll_mats[child].shape,"\n")
                #print(p_t[parent,child])
                LL_mat[parent] = p_t[parent,child].dot(ll_mats[child])
                #print(LL_mat[parent], LL_mat[parent].flags, ll_mats[child], ll_mats[child].flags, sep="\n")
            else:
                LL_mat[parent] *= p_t[parent,child].dot(ll_mats[child])
        else:
            if parent not in LL_mat:
                LL_mat[parent] = p_t[parent,child].dot(LL_mat[child])
                #print("parent", LL_mat[parent].flags, "child", LL_mat[child].flags, sep="\n")
            else:
                LL_mat[parent] *= p_t[parent,child].dot(LL_mat[child])
        #print(parent, LL_mat[parent])
    #print("Root ", root,LL_mat[root], LL_mat[root].shape,pi, pi.shape, sep="\n")
    #print(np.dot(pi, LL_mat[root]))
    #print(LL_mat[root].shape)
    ll = np.sum(np.log(np.dot(pi, LL_mat[root])))
    return ll, LL_mat

def matML_inplace_bl(state, taxa, ll_mats, old_LL_Mat, start_edge):
    LL_mat = defaultdict()
    root = state["root"]
    p_t = state["transitionMat"]
    pi = state["pi"]
    edges = state["postorder"]
    
    flag = False
    
    for edge in edges[::-1]:
        parent, child = edge

        if start_edge == edge:
            flag = True

        if not flag:
            LL_mat[parent] = old_LL_Mat[parent].copy()
            continue

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


