"""The data structures for this are based on adjaceny list or a graph data structure so that
iterative computing is possible. 
Refer: Valiente, Gabriel. Algorithms on trees and graphs. Springer Science & Business Media, 2013. for
further algorithms.
The tree datastructure also looks like "phylo" class in ape package of R.

author: Taraka Rama
email: taraka@fripost.org

"""

from collections import defaultdict, deque
import random
import profile
from operator import itemgetter
import numpy as np
random.seed(1234)
import itertools as it
np.random.seed(1234)
from scipy.stats import expon
import math
from libc.math cimport exp as c_exp



def adjlist2newick(nodes_dict, node, leaves):
    """Converts from edge list to NEWICK format.
    """
    tree_list = []
    
    x, y = nodes_dict[node]
    
    if x not in leaves:
        tree_list.append(adjlist2newick(nodes_dict, x, leaves))
    else:
        tree_list.append(x)
    if y not in leaves:
        tree_list.append(adjlist2newick(nodes_dict, y, leaves))
    else:
        tree_list.append(y)
    
    return "("+", ".join(map(str, tree_list))+")"



        
    #reverse_nodes_dict = defaultdict()
    #for edge in edges_list:
    #    reverse_nodes_dict[edge[1]] = edge[0]
    #return reverse_nodes_dict


cpdef get_children(edges_list, list leaves):
    """Get path from root to a tip"""
    cdef int root_node
    cdef dict root2paths
    cdef list paths
    
    root_node = 2*len(leaves) -1
    root2paths = {}
    for x, y in edges_list:
        root2paths[y] = x
    
    paths=[]
    for leaf in leaves:
        path = [leaf]
        while(1):
            leaf = root2paths[leaf]
            path += [leaf]
            if leaf == root_node:
                break
        paths.append(path)
    children = defaultdict(list)
    for path in paths:
        children[path[0]] = path[1:]
    return children

def get_subtree(edges_list, list leaves):
    """Get leaves under a specific node"""
    children =get_children(edges_list, leaves)
    subtree_dict = defaultdict(list)
    
    for k, v in children.items():
        for l in v:
            subtree_dict[l].append(k)
    return subtree_dict


def scale_all_edges(temp_edges_dict):
    n_edges = len(temp_edges_dict.keys())
    
    log_c = scaler_alpha*(random.uniform(0,1)-0.5)
    c = c_exp(log_c)
    
    prior_ratio = 0.0
    
    for edge, length in temp_edges_dict.items():
        temp_edges_dict[edge] = length*c
        #prior_ratio += expon.logpdf(length*c, scale=bl_exp_scale) - expon.logpdf(length, scale=bl_exp_scale)
        prior_ratio += -(length*c-length)/bl_exp_scale # Used the exponential distribution directly
    
    prior_ratio += n_edges*log_c
    
    return temp_edges_dict, prior_ratio


