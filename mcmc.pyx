import random
import numpy as np
cimport numpy as np

from scipy import linalg
np.random.seed(1234)

from scipy.stats import dirichlet

from ML import matML, cache_matML
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
from libc.stdlib cimport rand, RAND_MAX

import config

from multiprocessing import Pool

cdef double bl_exp_scale = 0.1
cdef double scaler_alpha = 1.0
cdef double epsilon = 1e-10


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

cpdef scale_edge(dict temp_edges_dict):
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
    rand_bl = bl_a+bl_b

    log_c = scaler_alpha*(random.random()-0.5)
    c = c_exp(log_c)
    rand_bl_new = rand_bl*c
    
    temp_edges_dict[parent_a, rand_edge[0]] = rand_bl_new*random.random()
    temp_edges_dict[rand_edge] = rand_bl_new - temp_edges_dict[parent_a, rand_edge[0]]

    #prior_ratio = expon.logpdf(rand_bl_new, scale=bl_exp_scale) - expon.logpdf(rand_bl, scale=bl_exp_scale)
    
    prior_ratio = -(rand_bl_new-rand_bl)/bl_exp_scale
    
    #prior_ratio = -math.log(bl_exp_scale*rand_bl_new) + math.log(bl_exp_scale*rand_bl)
    #prior_ratio = bl_exp_scale*(rand_bl-rand_bl_new)
    
    return temp_edges_dict, log_c, prior_ratio, rand_edge, (parent_a, rand_edge[0])

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
    
    return temp_edges_list, new_postorder, hastings_ratio, nodes_recompute

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
    
    leaf = random.randint(1, config.N_TAXA)
    
    parent_leaf = rev_nodes_dict[leaf]

    tgt = random.choice(list(edges_list))
    
    if parent_leaf == root_node or parent_leaf in tgt:
        hastings_ratio = 0.0
    elif rev_nodes_dict[parent_leaf] in tgt:
        hastings_ratio = 0.0
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
        
        u = random.random()
        edges_list[tgt[0],parent_leaf] = r*u
        edges_list[parent_leaf,tgt[1]] = r*(1.0-u)
        edges_list[rev_nodes_dict[parent_leaf], other_child_parent_leaf]=x+y
        hastings_ratio = r/(x+y)

        
    temp_nodes_dict = adjlist2nodes_dict(edges_list)
    new_postorder = postorder(temp_nodes_dict, root_node)[::-1]

    return edges_list, new_postorder, hastings_ratio

cpdef mvDualSlider(double[:] pi):
    cdef int i, j 
    i, j = random.sample(range(config.N_CHARS), 2)
    cdef double sum_ij = pi[i]+pi[j]
    cdef double x = random.uniform(epsilon, sum_ij)
    cdef double y = sum_ij -x
    pi[i], pi[j] = x, y
    
    
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
    which stores the information about children nodes.
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
    which stores the information about parent nodes.
    """
    cdef dict reverse_nodes_dict
    cdef int k
    reverse_nodes_dict = {v:k for k,v in edges_dict}
    #print(reverse_nodes_dict)
    return reverse_nodes_dict

cpdef init_tree():
    t = rtree()
    edge_dict, n_nodes = newick2bl(t)
    
    temp_edge_items = edge_dict.copy()
    
    for x, y in temp_edge_items:
        if y in config.TAXA:
            del edge_dict[x,y]
            edge_dict[x, config.TAXA.index(y)+1] = 1
    
    for k, v in edge_dict.items():
        edge_dict[k] = random.expovariate(1.0/bl_exp_scale)
    
    #print edge_dict
    
    return edge_dict, n_nodes

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
    #print(edges_dict)
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
    #cdef double[:] pi
    #cdef double[:,:] er
    #print config.N_CHARS
    
    if config.MODEL == "JC":
        pi = np.repeat(1.0/config.N_CHARS, config.N_CHARS)
    elif config.MODEL in ["F81", "GTR"]:
        pi = np.random.dirichlet(np.repeat(1,config.N_CHARS))
        #print pi
    er = np.random.dirichlet(np.repeat(1,config.N_CHARS*(config.N_CHARS-1)/2))
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

cpdef get_edge_transition_mat(pi, rates, double d, dict transition_mat, tuple change_edge):
    """Calcualtes new matrix and remembers the old matrix for a branch.
    """
    cdef int parent, child
    cdef double x, y
    
    if config.MODEL == "F81":
        config.NORM_BETA = 1/(1-np.dot(pi, pi))
      
    parent,child = change_edge
    
    if config.MODEL == "F81":
        x = c_exp(-config.NORM_BETA*d)
        y = 1.0-x

        if config.IN_DTYPE == "multi":
            transition_mat[parent,child] = ptF81(pi, x, y)
        elif config.IN_DTYPE == "bin":
            transition_mat[parent,child] = binaryptF81(pi, x, y)
    elif config.MODEL == "JC":
        x = c_exp(-config.NORM_BETA*d)
        y = (1.0-x)/config.N_CHARS
        transition_mat[parent,child] =  ptJC(x, y)

    return transition_mat

cpdef get_F81_prob(pi, edges_dict, move):
    cdef p_t = {}
    cdef double d, x, y
    cdef int parent, child
    config.NORM_BETA = 1/(1-np.dot(pi, pi))
    #print "NORM BETA ", config.NORM_BETA
    for parent, child in edges_dict:
        d = edges_dict[parent,child]
        x = c_exp(-config.NORM_BETA*d)
        y = 1.0-x
        #print x, y
        p_t[parent,child] = move(pi, x, y)
    return p_t

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

cpdef get_JC_prob(edges_dict, move):
    cdef p_t = {}
    cdef double d, x, y
    cdef int parent, child
    
    for parent, child in edges_dict:
        d = edges_dict[parent,child]
        x = c_exp(-config.NORM_BETA*d)
        y = (1.0-x)/config.N_CHARS
        p_t[parent,child] = move(x, y)
    return p_t

cpdef get_prob_t(pi, dict edges_dict, rates=None):
    if config.MODEL == "F81":
        if config.IN_DTYPE == "multi":
            return get_F81_prob(pi, edges_dict, ptF81)
        else:
            return get_F81_prob(pi, edges_dict, binaryptF81)
    elif config.MODEL == "JC":
        return get_JC_prob(edges_dict, ptJC)

cpdef ptJC(double x, double y):
    """Compute the Probability matrix under a F81 model
    """
    cdef np.ndarray p_t
    p_t = np.empty((config.N_CHARS, config.N_CHARS))
    p_t.fill(y)
    np.fill_diagonal(p_t, x+y)
    return p_t

cpdef binaryptF81(pi, double x, double y):
    """Compute the probability matrix for binary characters
    """
    cdef np.ndarray p_t
    p_t = np.empty((2,2))
    p_t[0, 0] = pi[0]+pi[1]*x
    p_t[0, 1] = pi[1]*y
    p_t[1, 0] = pi[0]*y
    p_t[1, 1] = pi[1]+pi[0]*x
    return p_t

cpdef ptF81(pi, double x, double y):
    """Compute the Probability matrix under a F81 model
    """
    cdef np.ndarray p_t
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
    nodes_dict = adjlist2nodes_dict(state["tree"])
    edges_ordered_list = postorder(nodes_dict, state["root"])[::-1]
    state["postorder"] = edges_ordered_list
    state["transitionMat"] = get_prob_t(state["pi"], state["tree"])
    return state


