import random, copy
import numpy as np
cimport numpy as np

from scipy import linalg
np.random.seed(1234)
random.seed(1234)
from scipy.stats import dirichlet

from scipy.special import gammainc
from scipy.stats import chi2, gamma

from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
from libc.stdlib cimport rand, RAND_MAX

import config

from multiprocessing import Pool

cdef double bl_exp_scale = 0.1
cdef double scaler_alpha = 1.0
cdef double epsilon = 1e-10
cdef double window = 0.1

cdef double min_shape_pr = 0.0001
cdef double max_shape_pr = 100
cdef double shape_pr_exp = 1.0
cdef double shape_scaler_tuning = 0.5
cdef double shape_slider_window = 0.5
cdef double a_S = 2
cdef double b_S = 4

cdef double min_branch_len = 0.00001
cdef double max_branch_len = 100
cdef double edge_slide_window = 0.1
cdef double edge_scaler_window = 1.2
cdef double a_T = 1
cdef double b_T = 0.1#For large phylogenies up to 1000

cpdef get_path2root(dict X, int internal_node, int root):
    cdef list paths = []
    cdef int parent
    #print("Internal node ", internal_node)
    while(1):
        parent = X[internal_node]
        #paths += [parent]
        paths.append(parent)
        internal_node = parent
        if parent == root:
            break
    
    return paths

cpdef scale_edge_exponential(dict temp_edges_dict):
    cdef tuple rand_edge
    cdef double rand_bl, rand_bl_new, log_c, c, prior_ratio
    
    rand_edge = random.choice(list(temp_edges_dict))

    rand_bl = temp_edges_dict[rand_edge]

    log_c = scaler_alpha*(random.random()-0.5)
    c = c_exp(log_c)
    rand_bl_new = rand_bl*c
    temp_edges_dict[rand_edge] = rand_bl_new

    #prior_ratio = expon.logpdf(rand_bl_new, scale=bl_exp_scale) - expon.logpdf(rand_bl, scale=bl_exp_scale)
    
    prior_ratio = -(rand_bl_new-rand_bl)/bl_exp_scale
    
    #prior_ratio = -math.log(bl_exp_scale*rand_bl_new) + math.log(bl_exp_scale*rand_bl)
    #prior_ratio = bl_exp_scale*(rand_bl-rand_bl_new)
    
    return temp_edges_dict, log_c, prior_ratio, rand_edge

cpdef node_slider_exponential(dict temp_edges_dict, int root_node):
    cdef tuple rand_edge
    cdef double rand_bl, rand_bl_new, log_c, c, prior_ratio
    
    nodes_dict = adjlist2reverse_nodes_dict(temp_edges_dict)
    
    while(1):
        rand_edge = random.choice(list(temp_edges_dict))
        if rand_edge[0] != root_node:
            break
    
    parent_a = nodes_dict[rand_edge[0]]
    bl_a = temp_edges_dict[parent_a, rand_edge[0]]
    bl_b = temp_edges_dict[rand_edge]
    rand_bl = bl_a+bl_b

    log_c = scaler_alpha*(np.random.random()-0.5)
    c = c_exp(log_c)
    rand_bl_new = rand_bl*c
    
    temp_edges_dict[parent_a, rand_edge[0]] = rand_bl_new*np.random.random()
    temp_edges_dict[rand_edge] = rand_bl_new - temp_edges_dict[parent_a, rand_edge[0]]

    #prior_ratio = expon.logpdf(rand_bl_new, scale=bl_exp_scale) - expon.logpdf(rand_bl, scale=bl_exp_scale)
    
    prior_ratio = -(rand_bl_new-rand_bl)/bl_exp_scale
    
    #prior_ratio = -math.log(bl_exp_scale*rand_bl_new) + math.log(bl_exp_scale*rand_bl)
    #prior_ratio = bl_exp_scale*(rand_bl-rand_bl_new)
    
    return temp_edges_dict, log_c, prior_ratio, rand_edge, (parent_a, rand_edge[0])

cpdef scale_edge(dict temp_edges_dict):
    cdef tuple rand_edge
    cdef double rand_bl, rand_bl_new, log_c, c, prior_ratio
    
    rand_edge = random.choice(list(temp_edges_dict))

    old_bl = temp_edges_dict[rand_edge]

    log_c = edge_scaler_window*(np.random.random()-0.5)
    c = c_exp(log_c)
    new_bl = old_bl*c

    while(1):
        if new_bl >= max_branch_len:
            new_bl = max_branch_len**2/new_bl
        elif new_bl <= min_branch_len:
            new_bl = min_branch_len**2/new_bl            
        elif new_bl < max_branch_len and new_bl > min_branch_len:
            break

    temp_edges_dict[rand_edge] = new_bl

    tl_old = sum(temp_edges_dict.values())
    tl_new = tl_old-old_bl+new_bl

    #prior_ratio = expon.logpdf(rand_bl_new, scale=bl_exp_scale) - expon.logpdf(rand_bl, scale=bl_exp_scale)
    
#    prior_ratio = -(rand_bl_new-rand_bl)/bl_exp_scale #exponential prior
    
    prior_ratio = -(b_T*(tl_new-tl_old)) -((2.0*config.N_TAXA-3)*(np.log(tl_new/tl_old)))#Gamma.Dirichlet prior with beta and alpha 1 and symmetric dirichlet
    
    #prior_ratio = -math.log(bl_exp_scale*rand_bl_new) + math.log(bl_exp_scale*rand_bl)
    #prior_ratio = bl_exp_scale*(rand_bl-rand_bl_new)

    proposal_ratio = np.log(new_bl/old_bl)    

    return temp_edges_dict, proposal_ratio, prior_ratio, rand_edge


cpdef slide_edge(dict temp_edges_dict):
    cdef tuple rand_edge
    cdef double rand_bl, rand_bl_new, log_c, c, prior_ratio
    
    rand_edge = random.choice(list(temp_edges_dict))

    old_bl = temp_edges_dict[rand_edge]

    new_bl = np.random.uniform(old_bl-edge_slide_window/2, old_bl+edge_slide_window/2)

    while(1):
        if new_bl >= max_branch_len:
            new_bl = 2*max_branch_len - new_bl
        elif new_bl <= min_branch_len:
            new_bl = 2*min_branch_len - new_bl
        elif new_bl > min_branch_len and new_bl < max_branch_len:
            break

    temp_edges_dict[rand_edge] = new_bl

    tl_old = sum(temp_edges_dict.values())
    tl_new = tl_old-old_bl+new_bl

    
    prior_ratio = -(b_T*(tl_new-tl_old)) -((2.0*config.N_TAXA-3)*(np.log(tl_new/tl_old)))#Gamma.Dirichlet prior with beta and alpha 1 and symmetric dirichlet

    proposal_ratio = 0.0

    return temp_edges_dict, proposal_ratio, prior_ratio, rand_edge


cpdef node_slider(dict temp_edges_dict, int root_node):
    cdef tuple rand_edge
    cdef double rand_bl, rand_bl_new, log_c, c, prior_ratio
    
    nodes_dict = adjlist2reverse_nodes_dict(temp_edges_dict)
    
    while(1):
        rand_edge = random.choice(list(temp_edges_dict))
        if rand_edge[0] != root_node:
            break
    
    parent_a = nodes_dict[rand_edge[0]]
    bl_a = temp_edges_dict[parent_a, rand_edge[0]]
    bl_b = temp_edges_dict[rand_edge]
#    old_bl = bl_a+bl_b

    log_c = scaler_alpha*(np.random.random()-0.5)
    c = c_exp(log_c)
#    new_bl = old_bl*c

    proposal_ratio = 2*(log_c + np.log(c*bl_a - min_branch_len) - np.log(bl_a - min_branch_len) + np.log(c*bl_b - min_branch_len) - np.log(bl_b - min_branch_len))

    temp_edges_dict[parent_a, rand_edge[0]] = c*bl_a
    temp_edges_dict[rand_edge] = c*bl_b

    tl_old = sum(temp_edges_dict.values())
    tl_new = tl_old + ((c-1)*(bl_a+bl_b))

#    while(1):
#        if new_bl >= max_branch_len:
#            new_bl = max_branch_len**2/new_bl
#        elif new_bl <= min_branch_len:
#            new_bl = min_branch_len**2/new_bl            
#        elif new_bl < max_branch_len and new_bl > min_branch_len:
#            break

#    temp_edges_dict[parent_a, rand_edge[0]] = new_bl*np.random.random()
#    temp_edges_dict[rand_edge] = new_bl - temp_edges_dict[parent_a, rand_edge[0]]

#    tl_old = sum(temp_edges_dict.values())
#    tl_new = tl_old-old_bl+new_bl

    #prior_ratio = expon.logpdf(rand_bl_new, scale=bl_exp_scale) - expon.logpdf(rand_bl, scale=bl_exp_scale)
    
#    prior_ratio = -(rand_bl_new-rand_bl)/bl_exp_scale#exponential prior

    prior_ratio = -(b_T*(tl_new-tl_old)) -((2.0*config.N_TAXA-3)*(np.log(tl_new/tl_old)))#Gamma.Dirichlet prior with beta =0.1 and alpha 1 and symmetric dirichlet

    #prior_ratio = -math.log(bl_exp_scale*rand_bl_new) + math.log(bl_exp_scale*rand_bl)
    #prior_ratio = bl_exp_scale*(rand_bl-rand_bl_new)
    
#    proposal_ratio = np.log(new_bl/old_bl) 

    return temp_edges_dict, proposal_ratio, prior_ratio, rand_edge, (parent_a, rand_edge[0])


cpdef rooted_NNI(dict temp_edges_list, int root_node):
    """Performs Nearest Neighbor Interchange on a edges list.
    """
    cdef double hastings_ratio = 0.0
    cdef double tgt_bl, src_bl
    cdef int a, b, src, tgt
    cdef list new_postorder
    cdef list nodes_recompute, x, y
    cdef dict temp_nodes_dict
    cdef dict nodes_dict
    cdef list shuffle_keys
    
    nodes_dict = adjlist2nodes_dict(temp_edges_list)
    shuffle_keys = list(temp_edges_list.keys())
    random.shuffle(shuffle_keys)
    for a, b in shuffle_keys:
        if b > config.N_TAXA:# and a != root_node:
            x, y = nodes_dict[a], nodes_dict[b]
            break
    #print("selected NNI ", a,b)
    #print("leaves ", x, y)
    if x[0] == b: src = x[1]
    else: src = x[0]
    tgt = random.choice(y)
    src_bl, tgt_bl = temp_edges_list[a, src], temp_edges_list[b, tgt]
    del temp_edges_list[a,src], temp_edges_list[b, tgt]
    temp_edges_list[a, tgt] = tgt_bl
    temp_edges_list[b, src] = src_bl
    
    temp_nodes_dict = adjlist2nodes_dict(temp_edges_list)
    new_postorder = postorder(temp_nodes_dict, root_node)[::-1]
    nodes_recompute = [b]+get_path2root(adjlist2reverse_nodes_dict(temp_edges_list), b, root_node)
    
    return temp_edges_list, new_postorder, hastings_ratio, nodes_recompute, [a,b,src, tgt]

cpdef externalSPR(dict edges_list,int root_node):
    """Performs Subtree-Pruning and Regrafting of an branch connected to terminal leaf
    """
    cdef double hastings_ratio, x, y, r, u
    cdef int leaf
    cdef int parent_leaf
    cdef tuple tgt
    cdef list new_postorder
    cdef dict temp_nodes_dict
    
    rev_nodes_dict = adjlist2reverse_nodes_dict(edges_list)
    nodes_dict = adjlist2nodes_dict(edges_list)
    
    #print("\n##### Old dictionary ########\n",nodes_dict,"\n")

#    while(1):
    leaf = random.randint(1, config.N_TAXA)
    
    parent_leaf = rev_nodes_dict[leaf]

    tgt = random.choice(list(edges_list))
    
    if parent_leaf == root_node or parent_leaf in tgt:
        hastings_ratio = 0.0
#        continue
    elif rev_nodes_dict[parent_leaf] in tgt:
        hastings_ratio = 0.0
#        continue
    else:
        children_parent_leaf = nodes_dict[parent_leaf]
        other_child_parent_leaf = children_parent_leaf[0]
        if leaf == other_child_parent_leaf:
            other_child_parent_leaf = children_parent_leaf[1]
        
        x = edges_list[rev_nodes_dict[parent_leaf], parent_leaf]
        y = edges_list[parent_leaf, other_child_parent_leaf]
        r = edges_list[tgt]
        
        del edges_list[rev_nodes_dict[parent_leaf], parent_leaf]
        del edges_list[parent_leaf, other_child_parent_leaf]
        del edges_list[tgt]
        
        u = np.random.random()
        edges_list[tgt[0],parent_leaf] = r*u
        edges_list[parent_leaf,tgt[1]] = r*(1.0-u)
        edges_list[rev_nodes_dict[parent_leaf], other_child_parent_leaf]=x+y
        hastings_ratio = r/(x+y)
#            break

#    assert hastings_ratio > 0
       
    temp_nodes_dict = adjlist2nodes_dict(edges_list)
    new_postorder = postorder(temp_nodes_dict, root_node)[::-1]

    return edges_list, new_postorder, hastings_ratio

#cpdef mvDualSlider(double[:] pi):
#    cdef int i, j
#    cdef double sum_ij = 0.0
#    cdef double low, high

#    i, j = random.sample(range(pi.shape[0]), 2)
#    sum_ij = pi[i]+pi[j]

#    low, high = pi[i], pi[j]
#    if pi[i] > pi[j]:
#        low = pi[j]
#        high = pi[i]

#    cdef double x = np.random.uniform(low, high)
#    pi[i] = x
#    pi[j] = sum_ij - x

#    cdef double y = sum_ij -x
#    pi[i], pi[j] = x, y

#    cdef double x = random.uniform(low, sum_ij)
#    cdef double y = sum_ij -x
    
#    cdef double x = sum_ij*random.uniform(epsilon, sum_ij)
#    cdef double y = sum_ij -x


#    x = random.uniform(pi[i]-window/2, pi[i]+window/2)

#    print("mvDualSlider ", pi[i], pi[j], sum_ij, x, pi[i], pi[j])

#    if x < 0:
#        print("Negative ", x)
#        x = -1*x
#        print("Negative ", x)
#    elif x > sum_ij:
#        x = 2*sum_ij-x

#    pi[i] = x
#    pi[j] = sum_ij-x

#    print("mvDualSlider ", pi[i], pi[j], sum_ij, x, pi[i], pi[j])


#    i, j = random.sample(range(config.N_CHARS), 2)

#    cdef double sum_ij = pi[i]+pi[j]
#    #cdef double x = random.uniform(epsilon, sum_ij)
#    cdef double x = sum_ij*random.random()
#    cdef double y = sum_ij -x
#    pi[i], pi[j] = x, y
        
#    return pi, 0.0

#cpdef mvRatesSlider(double[:] pi):
#    cdef int i, j 
#    i, j = random.sample(range(pi.shape[0]), 2)
##    i, j = random.sample(range(config.N_CHARS), 2)

#    cdef double sum_ij = pi[i]+pi[j]
#    #cdef double x = random.uniform(epsilon, sum_ij)
#    cdef double x = sum_ij*random.random()
#    cdef double y = sum_ij -x
#    pi[i], pi[j] = x, y
#        
#    return pi, 0.0

#cpdef mvSlider(double[:] pi):
#    x = np.random.uniform(pi[0]-window/2, pi[0]+window/2)
#    while(1):
#        if x <= 0:
#            x = -x
#        elif x >= 1:
#            x = 2-x
#        elif x > 0 and x < 1:
#            break


cpdef mvRatesSlider(double[:] pi):
    cdef int i, j
    i, j = random.sample(range(pi.shape[0]), 2)
    cdef double sum_ij = 0.0
    sum_ij = pi[i]+pi[j]
    
    x = np.random.uniform(pi[i]-window/2, pi[i]+window/2)

    while(1):
        if x <= 0:
            x = -x
        elif x >= sum_ij:
            x = 2*sum_ij-x
        elif x > 0 and x < sum_ij:
            break

    pi[i] = x
    pi[j] = sum_ij - x

#    print("rates",i, " = ", pi[i], "rates",j, " = ", pi[j])

    return pi, 0.0

cpdef scale_alpha(float alpha):
    log_c = scaler_alpha*(random.random()-0.5)
    c = c_exp(log_c)
    new_alpha = alpha*c
    pr_ratio = -(new_alpha-alpha)
    return new_alpha, log_c, pr_ratio


cpdef mvShapeScaler(float old_alpha):
    log_c = shape_scaler_tuning*(np.random.random()-0.5)
    c = c_exp(log_c)
    new_alpha = old_alpha*c

    while(1):
        if new_alpha >= max_shape_pr:
            new_alpha = max_shape_pr**2/new_alpha
        elif new_alpha <= min_shape_pr:
            new_alpha = min_shape_pr**2/new_alpha            
        elif new_alpha < max_shape_pr and new_alpha > min_shape_pr:
            break

#    pr_ratio = -shape_pr_exp*(new_alpha-old_alpha)

    pr_ratio = ((a_S-1)*np.log(new_alpha/old_alpha)) - (b_S *(new_alpha-old_alpha)) #A Gamma prior for discrete gamma shape prior

    proposal_ratio = np.log(new_alpha/old_alpha)

    #print("In mvShapeScaler new_alpha = ",new_alpha, "Old alpha = ", old_alpha)

    return new_alpha, proposal_ratio, pr_ratio

cpdef mvShapeSlider(float old_alpha):
    new_alpha = np.random.uniform(old_alpha-shape_slider_window/2, old_alpha+shape_slider_window/2)

    while(1):
        if new_alpha >= max_shape_pr:
            new_alpha = 2*max_shape_pr - new_alpha
        elif new_alpha <= min_shape_pr:
            new_alpha = 2*min_shape_pr - new_alpha
        elif new_alpha > min_shape_pr and new_alpha < max_shape_pr:
            break

#    pr_ratio = -shape_pr_exp*(new_alpha-old_alpha)

    pr_ratio = ((a_S-1)*np.log(new_alpha/old_alpha)) - (b_S *(new_alpha-old_alpha)) #A Gamma prior for discrete gamma shape prior

    proposal_ratio = 0

#    print("In mvShapeSlider new_alpha = ",new_alpha, "Old alpha = ", old_alpha)

    return new_alpha, proposal_ratio, pr_ratio

cpdef mvDualSlider(double[:] pi):
    cdef int i, j
    i, j = random.sample(range(pi.shape[0]), 2)
    cdef double sum_ij = 0.0
    sum_ij = pi[i]+pi[j]
    
    x = np.random.uniform(pi[i]-window/2, pi[i]+window/2)

    while(1):
        if x <= 0:
            x = -x
        elif x >= sum_ij:
            x = 2*sum_ij-x
        elif x > 0 and x < sum_ij:
            break

    pi[i] = x
    pi[j] = sum_ij - x

#    print("state",i, " = ", pi[i], "state",j, " = ", pi[j])

    return pi, 0.0

cpdef mvBinaryDualSlider(double[:] pi):
    cdef double x = 0.0

    x = np.random.uniform(pi[0]-window/2, pi[0]+window/2)
    while(1):
        if x <= 0:
            x = -x
        elif x >= 1:
            x = 2-x
        elif x > 0 and x < 1:
            break

    pi[0] = x
    pi[1] = 1.0 - x

#    print("state 0 = ", pi[0], "state 1 = ", pi[1])

#    sum_ij = pi[0]+pi[1]


#    cdef int i, j
#    cdef double sum_ij = pi[0]+pi[1]
#    #cdef double x = random.uniform(epsilon, sum_ij)
#    cdef double x = sum_ij*random.random()
#    cdef double y = sum_ij -x
#    pi[0], pi[1] = x, y

    return pi, 0.0


cpdef postorder(dict nodes_dict, int node):
    """Return the post-order of edges to be processed.
    """
    cdef list edges_ordered_list = []
    cdef int x, y
    #print node, nodes_dict[node]
    x, y = nodes_dict[node]
    #print node, x, y
    edges_ordered_list.append((node,x))
    edges_ordered_list.append((node,y))
    
    if x > config.N_TAXA:
        #z = postorder(nodes_dict, x, leaves)
        edges_ordered_list += postorder(nodes_dict, x)
    if y > config.N_TAXA:
        #w = postorder(nodes_dict, y, leaves)
        edges_ordered_list += postorder(nodes_dict, y)
    
    return edges_ordered_list

cpdef adjlist2nodes_dict(dict edges_dict):
    """Converts a adjacency list representation to a nodes dictionary
    which stores the information about neighboring nodes.
    """
    cdef tuple edge
    cdef dict nodes_dict = {}

    for edge in edges_dict:
        if edge[0] not in nodes_dict:
            nodes_dict[edge[0]] = [edge[1]]
        else:
            nodes_dict[edge[0]].append(edge[1])
    return nodes_dict

cpdef adjlist2reverse_nodes_dict(edges_dict):
    """Converts a adjacency list representation to a nodes dictionary
    which stores the information about neighboring nodes.
    """
    cdef dict reverse_nodes_dict
    cdef int k
    reverse_nodes_dict = {v:k for k,v in edges_dict}
    #print(reverse_nodes_dict)
    return reverse_nodes_dict

#cpdef init_tree(float b_T):
#    t = rtree()
#    edge_dict, n_nodes = newick2bl(t)
#    
#    temp_edge_items = edge_dict.copy()
#    
#    for x, y in temp_edge_items:
#        if y in config.TAXA:
#            del edge_dict[x,y]
#            edge_dict[x, config.TAXA.index(y)+1] = 1
#    
##    TL = random.gammavariate(a_T, b_T)

#    n_branches = 2*config.N_TAXA-2
##    TL = 0.5*n_branches
##    print("Initial tree length", TL, "with inverse beta parameter", b_T)

##    n_brans_lens = TL*np.random.dirichlet(np.repeat(1,n_branches))
#    n_brans_lens = [0.1]*n_branches
##    print("Tree length ", sum(n_brans_lens))
#    for i, k in enumerate(edge_dict.keys()):
#        edge_dict[k] = 0.2
##        edge_dict[k] = n_brans_lens[i]

#    print("Tree length ", sum(edge_dict.values()))


##    for k, v in edge_dict.items():
##        edge_dict[k] = random.expovariate(1.0/bl_exp_scale)

#    
#    #print edge_dict
#    
#    return edge_dict, n_nodes

cpdef init_tree():
    t = rtree()
    edge_dict, n_nodes = newick2bl(t)
    
    temp_edge_items = edge_dict.copy()
    
    for x, y in temp_edge_items:
        if y in config.TAXA:
            del edge_dict[x,y]
            edge_dict[x, config.TAXA.index(y)+1] = 1
    
#    TL = random.gammavariate(a_T, b_T)
    TL = np.random.gamma(a_T, scale=1/b_T)


    n_branches = 2*config.N_TAXA-2
#    print("Initial tree length", TL, "with inverse beta parameter", b_T)

    n_brans_lens = TL*np.random.dirichlet(np.repeat(1,n_branches))
    print("Tree length ", sum(n_brans_lens))
    for i, k in enumerate(edge_dict.keys()):
        edge_dict[k] = n_brans_lens[i]
#        edge_dict[k] = 0.1        

    print("Tree length from edge dict", sum(edge_dict.values()))


#    for k, v in edge_dict.items():
#        edge_dict[k] = random.expovariate(1.0/bl_exp_scale)

    
    #print edge_dict
    
    return edge_dict, n_nodes


cpdef init_alpha_rate():
#    x = np.random.exponential(1/shape_pr_exp)

#    while(1):
#        if x > min_shape_pr and x < max_shape_pr:
#            break
#        else:
#            x = np.random.exponential(1/shape_pr_exp)

    while(1):
        x = np.random.gamma(a_S, scale=1/b_S)

        if x > min_shape_pr and x < max_shape_pr:
            break

    return x
    
cpdef newick2bl(t):
    """Implement a function that can read branch lengths from a newick tree
    """
    n_leaves = len(t.split(","))

    n_internal_nodes = n_leaves+t.count("(")
    n_nodes = n_leaves+t.count("(")
    edges_dict = {}# defaultdict()
    t = t.replace(";","")
    t = t.replace(" ","")
    t = t.replace(")",",)")
    t = t.replace("(","(,")

    nodes_stack = []
    
    arr = t.split(",")

    for i, elem in enumerate(arr[:-1]):

        if "(" in elem:

            nodes_stack.append(n_internal_nodes)
            n_internal_nodes -= 1

        elif "(" not in elem and ")" not in elem:
            if ":" not in elem:
                k, v =elem, 1
            else:
                k, v = elem.split(":")
            edges_dict[nodes_stack[-1], k] = float(v)
        elif ")" in elem:
            if ":" not in elem:
                v = 1
            else:
                k, v = elem.split(":")
            k = nodes_stack.pop()
            edges_dict[nodes_stack[-1], k] = float(v)
    
    return edges_dict, n_nodes

cpdef rtree():
    """Generates random Trees
    """
    taxa_list = [t for t in config.TAXA]
    random.shuffle(taxa_list)
    while(len(taxa_list) > 1):
        ulti_elem = str(taxa_list.pop())
        penulti_elem = str(taxa_list.pop())
        taxa_list.insert(0, "(" + penulti_elem + "," + ulti_elem + ")")
        random.shuffle(taxa_list)

    taxa_list.append(";")
    return "".join(taxa_list)

cpdef init_pi_er():
    cdef double[:] pi, er
    #cdef double[:] er
    #print config.N_CHARS
    
    if config.MODEL == "JC":
        pi = np.repeat(1.0/config.N_CHARS, config.N_CHARS)
    elif config.MODEL in ["F81", "GTR"]:
        pi = np.random.dirichlet(np.repeat(1,config.N_CHARS))
        #print pi
    er = np.random.dirichlet(np.repeat(1,config.N_CHARS*(config.N_CHARS-1)/2))
    #print "Rates"
    #print er
    return pi, er

cpdef init_pi_er_ml():
    cdef double[:] pi, er
    #cdef double[:] er
    #print config.N_CHARS
    
    if config.MODEL == "JC":
        pi = np.repeat(1.0/config.N_CHARS, config.N_CHARS)
    elif config.MODEL in ["F81", "GTR"]:
        
        pi = np.random.dirichlet(np.repeat(1,config.N_CHARS))
        #print pi
    er = np.random.dirichlet(np.repeat(1,config.N_CHARS*(config.N_CHARS-1)/2))
    #print "Rates"
    #print er
    return pi, er


#def prior_probs(param, val):
#    if param == "pi":
#        return dirichlet.logpdf(val, alpha=prior_pi)
#    elif param == "rates":
#        return dirichlet.logpdf(val, alpha=prior_er)

cpdef get_copy_transition_mat(pi, rates, dict edges_dict,dict transition_mat,tuple change_edge):
    cdef tuple Edge
    cdef int parent, child
    cdef double d, x, y
    
    if config.MODEL == "F81":
        config.NORM_BETA = 1/(1-np.dot(pi, pi))
        
    cdef dict new_transition_mat = {}
    
    for Edge in edges_dict:
        parent, child = Edge
        if Edge != change_edge:
            new_transition_mat[parent, child] = transition_mat[parent, child].copy()
        else:
            d = edges_dict[parent,child]
            if config.MODEL == "F81":
                x = c_exp(-config.NORM_BETA*d)
                y = 1.0-x

                if config.IN_DTYPE == "multi":
                    new_transition_mat[parent,child] = ptF81(pi, x, y)
                elif config.IN_DTYPE == "bin":
                    new_transition_mat[parent,child] = binaryptF81(pi, x, y)
            elif config.MODEL == "JC":
                x = c_exp(-config.NORM_BETA*d)
                y = (1.0-x)/config.N_CHARS
                #p_t[parent,child] =  subst_models.fastJC(n_chars, x, y)
                new_transition_mat[parent,child] =  ptJC(x, y)
    return new_transition_mat


cpdef get_edge_transition_mat(double[:] pi, double[:] rates, double d):
    """Calculates new matrix and remembers the old matrix for a branch.
    """
    cdef int parent, child
    cdef double x, y
    
    if config.MODEL == "F81":
        config.NORM_BETA = 1/(1-np.dot(pi, pi))
      
    #parent,child = change_edge
    
    if config.MODEL == "F81":
        x = c_exp(-config.NORM_BETA*d)
        y = 1.0-x
        if config.IN_DTYPE == "multi":
            return ptF81(pi, x, y)
            #transition_mat[parent,child] = ptF81(pi, x, y)
        elif config.IN_DTYPE == "bin":
            return binaryptF81(pi, x, y)
            #transition_mat[parent,child] = binaryptF81(pi, x, y)
    elif config.MODEL == "JC":
        x = c_exp(-config.NORM_BETA*d)
        y = (1.0-x)/config.N_CHARS
        #transition_mat[parent,child] =  ptJC(x, y)
        return ptJC(x, y)
    elif config.MODEL == "GTR":
        Q = fnGTR(rates, pi)
        #transition_mat[parent,child] = linalg.expm(Q*d)
        return linalg.expm(Q*d)
    #return transition_mat

cpdef get_transition_mat_NNI(dict tmat, list nodes_list):
    """Calculates new matrix and remembers the old matrix for a branch.
    """
    cdef int a, b, src, tgt

    a, b, src, tgt = nodes_list
    #p_t_a_src, p_t_b_tgt = tmat[a,src], tmat[b,tgt]
        
    #tmat[a,tgt], tmat[b,src] = np.copy(tmat[b,tgt], order='K'), np.copy(tmat[a,src], order='K')
    tmat[a,tgt], tmat[b,src] = copy.deepcopy(tmat[b,tgt]), copy.deepcopy(tmat[a,src])
    
    #del tmat[a,src], tmat[b,tgt]

    return tmat        

cpdef par_get_JC_prob(edges_dict, move):
    cdef p_t = {}
    cdef double d, x, y
    cdef int parent, child
    
    p = Pool(2)
    keys, values= zip(*edges_dict.items())
    X = np.exp(-config.NORM_BETA*np.array(values))
    Y = 1.0-X
    Y /= config.N_CHARS
    proc_values = p.starmap(ptJC, zip(X,Y),chunksize=50)
    
    p_t = dict(zip(keys, proc_values))
    p.close()
    #for parent, child in edges_dict:
    #    d = edges_dict[parent,child]
    #    x = c_exp(-config.NORM_BETA*d)
    #    y = (1.0-x)/config.N_CHARS
    #    p_t[parent,child] = move(x, y)
    return p_t

cpdef get_prob_t(double[:] pi, dict edges_dict, double[:] rates, double mean_rate):
    if config.MODEL == "F81":
        if config.IN_DTYPE == "multi":
            return get_F81_prob(pi, edges_dict, ptF81, mean_rate)
        else:
            return get_F81_prob(pi, edges_dict, binaryptF81, mean_rate)
    elif config.MODEL == "JC":
        return get_JC_prob(edges_dict, ptJC, mean_rate)
    elif config.MODEL == "GTR":
        return get_GTR_prob(pi, rates, edges_dict, mean_rate)

cpdef get_JC_prob(dict edges_dict, move, double mean_rate):
    cdef dict p_t = {}
    cdef double d, x, y, v
    cdef tuple k
    
    for k, v in edges_dict.items():
        d = v*mean_rate
        x = c_exp(-config.NORM_BETA*d)
        y = (1.0-x)/config.N_CHARS
        p_t[k] = move(x, y)
    return p_t

cpdef get_F81_prob(double[:] pi, dict edges_dict, move, double mean_rate):
    cdef dict p_t = {}
    cdef double d, x, y, v
    cdef tuple k
    #cdef int parent, child
    config.NORM_BETA = 1/(1-np.dot(pi, pi))
    #print "NORM BETA ", config.NORM_BETA
    for k, v in edges_dict.items():
        d = v*mean_rate
        x = c_exp(-config.NORM_BETA*d)
        y = 1.0-x
        #print x, y
        p_t[k] = move(pi, x, y)
    return p_t

cpdef get_GTR_prob(pi, rates, edges_dict, mean_rate):
    Q = fnGTR(rates, pi)
    cdef p_t = {} 
    for parent, child in edges_dict:
        p_t[parent,child] = linalg.expm(Q*edges_dict[parent,child]*mean_rate)
    return p_t

cpdef fnGTR(er, pi):
    n_states = pi.shape[0]
    n_rates = er.shape[0]
    R = np.zeros((n_states, n_states))
    iu1 = np.triu_indices(n_states,1)
    #il1 = np.tril_indices(n_states,-1)
    R[iu1] = er
    R = R+R.T
    #R[il1] = er

    Pi = np.diag(pi)
    Q = np.dot(R,Pi)
    
    #X = np.diag(-np.dot(pi,R)/pi)
    #R = R + X

    Q += np.diag(-np.sum(Q,axis=-1))
    beta = -1.0/np.dot(pi,np.diag(Q))
    Q = Q*beta
    #print("\n")
    #print(np.dot(pi,Q))
    return Q

cpdef ptJC(double x, double y):
    """Compute the Probability matrix under a JC model
    """
    cdef np.ndarray[double, ndim=2] p_t
    p_t = np.empty((config.N_CHARS, config.N_CHARS))
    p_t.fill(y)
    np.fill_diagonal(p_t, x+y)
    return p_t

cpdef binaryptF81(double[:] pi, double x, double y):
    """Compute the probability matrix for binary characters
    """
    cdef np.ndarray[double, ndim=2] p_t
    p_t = np.empty((2,2))
    p_t[0, 0] = pi[0]+pi[1]*x
    p_t[0, 1] = pi[1]*y
    p_t[1, 0] = pi[0]*y
    p_t[1, 1] = pi[1]+pi[0]*x
    return p_t

cpdef ptF81(double[:] pi, double x, double y):
    """Compute the Probability matrix under a F81 model
    """
    cdef np.ndarray[double, ndim=2] p_t
    #print pi, x, y
    p_t = np.empty((config.N_CHARS, config.N_CHARS))
    cdef int i, j
    for i in range(config.N_CHARS):
        for j in range(config.N_CHARS):
            if i==j:
                p_t[i,j] = pi[i]*y+x
            else:
                p_t[i,j] = pi[j]*y

    #for i in range(config.N_CHARS):
    #    p_t[i] = pi*y
    #p_t += np.eye(config.N_CHARS)*x
    #p_t = np.tile(pi*y,(config.N_CHARS, 1)) + np.eye(config.N_CHARS)*x

    #p_t = np.array([pi*y]*config.N_CHARS)+np.eye(config.N_CHARS)*x
    return p_t

cpdef adjlist2newickBL(dict edges_list, dict nodes_dict, int node):
    """Converts from edge list to NEWICK format.
    """
    cdef list tree_list = []    
    cdef int x, y
    
    x, y = nodes_dict[node]
    #print(node, x, y)
    if x > config.N_TAXA:
        #print x, edges_list[node,x]
        tree_list.append(adjlist2newickBL(edges_list, nodes_dict, x)+":"+str(edges_list[node,x]))
    else:
        #print x, edges_list[node,x]
        tree_list.append(config.TAXA[x-1]+":"+str(edges_list[node,x]))

    if y > config.N_TAXA:
        #print(y)
        tree_list.append(adjlist2newickBL(edges_list, nodes_dict, y)+":"+str(edges_list[node,y]))
    else:
        #print(y, edges_list[node,y])
        tree_list.append(config.TAXA[y-1]+":"+str(edges_list[node,y]))
    #print(tree_list)
    return "("+", ".join(map(str, tree_list))+")"

cpdef state_init():
    cdef dict state = {}
    cdef dict nodes_dict
    cdef list edges_ordered_list
    #print "Initializing states ", config.N_CHARS
    pi, er = init_pi_er()
    
    config.NORM_BETA = 1/(1-np.dot(pi, pi))
    
    state["pi"] = pi
    state["rates"] = er
    state["tree"], state["root"] = init_tree()
    #state["srates"] = 1
    state["srates"] = init_alpha_rate()#. Fixing this to avoid NaN issues when generating site rates using Chi-Square distribution
    nodes_dict = adjlist2nodes_dict(state["tree"])
    edges_ordered_list = postorder(nodes_dict, state["root"])[::-1]
    state["postorder"] = edges_ordered_list
    site_rates = get_siterates(state["srates"])
    state["transitionMat"] = []
    for mean_rate in site_rates:
        state["transitionMat"].append(get_prob_t(state["pi"], state["tree"], state["rates"], mean_rate))
    return state


cpdef get_siterates(float alpha):

    if config.N_CATS == 1: return [1]

    site_rates = [0]*config.N_CATS

    cutoffs = [chi2.isf(1-p,2*alpha)/(2*alpha) for p in np.arange(0,1,1.0/config.N_CATS)]
#    print("Chi Cutoffs =", cutoffs)

    for i in range(1,config.N_CATS):
#        print(i, cutoffs[i-1], cutoffs[i], (gammainc(alpha+1,cutoffs[i]*alpha)-gammainc(alpha+1,cutoffs[i-1]*alpha))*N_CATS)
        site_rates[i-1] = (gammainc(alpha+1,cutoffs[i]*alpha)-gammainc(alpha+1,cutoffs[i-1]*alpha))*config.N_CATS
#    print(i+1, cutoffs[-1], (1.0-gammainc(alpha+1,cutoffs[-1]*alpha))*N_CATS)

    site_rates[-1] = (1.0-gammainc(alpha+1,cutoffs[-1]*alpha))*config.N_CATS

    return site_rates

#    cutoffs = [chi2.isf(1-p,2*alpha)/(2*alpha) for p in np.arange(1.0/config.N_CATS,1,1.0/config.N_CATS)]
#    if config.N_CATS == 3:
#        cutoffs = cutoffs[:-1]
#    site_rates = [gammainc(alpha+1,cutoffs[0]*alpha)*config.N_CATS]
#    for i in range(1,config.N_CATS-1):
#        site_rates.append((gammainc(alpha+1,cutoffs[i]*alpha)-gammainc(alpha+1,cutoffs[i-1]*alpha))*config.N_CATS)
#    site_rates.append((1.0-gammainc(alpha+1,cutoffs[-1]*alpha))*config.N_CATS)
#    return site_rates

#cpdef get_siterates(float alpha):
##    cutoffs = [chi2.isf(1-p,2*alpha)/(2*alpha) for p in np.arange(0,1,1.0/config.N_CATS)]
#    cutoffs = [chi2.isf(1-p,2*alpha) for p in np.arange(0,1,1.0/config.N_CATS)]
##    cutoffs = [gamma.ppf(p,alpha, loc=0, scale=1.0/alpha) for p in np.arange(0,1,1.0/config.N_CATS)]

#    if config.N_CATS == 1: return [1]

#    site_rates = [0]*config.N_CATS

#    for i in range(1,config.N_CATS):
#        site_rates[i-1] = (gammainc(alpha+1,cutoffs[i]*alpha)-gammainc(alpha+1,cutoffs[i-1]*alpha))*config.N_CATS

#    site_rates[-1] = (1.0-gammainc(alpha+1,cutoffs[-1]*alpha))*config.N_CATS

#    return site_rates



