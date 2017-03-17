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
random.seed(12345)
import itertools as it
np.random.seed(1234)
from scipy.stats import expon
bl_exp_scale = 0.1
scaler_alpha = 1.25

def init_tree(taxa):
    t = rtree(taxa)
    edge_dict, n_nodes = newick2bl(t)
    
    for k, v in edge_dict.items():
        edge_dict[k] = np.random.exponential(bl_exp_scale)
    
    return edge_dict, n_nodes

def newick2bl(t):
    """Implement a function that can read branch lengths from a newick tree
    """
    n_leaves = len(t.split(","))
    #assert n_taxa == n_leaves
    n_internal_nodes = n_leaves+t.count("(")
    n_nodes = n_leaves+t.count("(")
    edges_dict = defaultdict()
    t = t.replace(";","")
    t = t.replace(" ","")
    t = t.replace(")",",)")
    t = t.replace("(","(,")

    nodes_stack = []
    
    arr = t.split(",")
    #print(arr)
    for i, elem in enumerate(arr[:-1]):
        #print("Before ",nodes_stack)
        if "(" in elem:
            #k, v = elem.replace("(","").split(":")
            nodes_stack.append(n_internal_nodes)
            n_internal_nodes -= 1
            #edges_dict[nodes_stack[-1], k] = float(v)
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
        #print("After ",nodes_stack)
        #print("EDGES ",edges_dict)
    #print("EDGES ",edges_dict, "\n")
    
    return edges_dict, n_nodes

def newick2adjlist(t, n_taxa):
    """Converts from a NEWICK format to a adjacency list format.
    Allows fast computation for Parsimony.
    """
    n_leaves = len(t.split(","))
    assert n_taxa == n_leaves
    leaves = []
    nodes_stack = []
    leaf = ""
    
    n_internal_nodes = 2*n_leaves-1
    t = t.replace(";","")
    t = t.replace(" ","")
    
    edges_list = defaultdict()
    #edges_list = []
    last_internal_node_idx = []
    for i, ch in enumerate(t):
        if ch == "(":
            nodes_stack.append(n_internal_nodes)
            n_internal_nodes -= 1
            last_internal_node_idx.append(len(nodes_stack)-1)
        elif ch == " ":
            continue
        elif ch == ")":
            if leaf != None:
                nodes_stack.append(leaf)
                leaves.append(leaf)
            leaf = None
            last_internal_node = nodes_stack[last_internal_node_idx[-1]]
            sub_nodes = [x for x in nodes_stack[last_internal_node_idx[-1]+1:]]
            for x in sub_nodes:
                #edges_list.append((last_internal_node, x))
                edges_list[last_internal_node,x] = 1
                nodes_stack.pop()
            last_internal_node_idx.pop()
        elif ch == ",":
            if t[i-1] != ")":
                nodes_stack.append(leaf)
                leaves.append(leaf)
                leaf = None
            else:
                continue
        else:
            if leaf == None:
                leaf = ch
            else:
                leaf += ch
    
    return (edges_list, leaves)

def rtree(taxa):
    """Generates random Trees
    """
    taxa_list = [t for t in taxa]
    random.shuffle(taxa_list)
    while(len(taxa_list) > 1):
        ulti_elem = str(taxa_list.pop())
        penulti_elem = str(taxa_list.pop())
        taxa_list.insert(0, "(" + penulti_elem + "," + ulti_elem + ")")
        random.shuffle(taxa_list)

    taxa_list.append(";")
    return "".join(taxa_list)

def postorder(nodes_dict, node, leaves):
    """Return the post-order of edges to be processed.
    """
    edges_ordered_list = []
    #print(node, nodes_dict[node])
    x, y = nodes_dict[node]
    #print(node, x, y)
    edges_ordered_list += [(node,x)]
    edges_ordered_list += [(node,y)]
    
    if x not in leaves:
        z = postorder(nodes_dict, x, leaves)
        edges_ordered_list += z
    if y not in leaves:
        w = postorder(nodes_dict, y, leaves)
        edges_ordered_list += w
    
    return edges_ordered_list

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

def adjlist2NewickNames(nodes_dict, node, leaves, lang_dict):
    """Converts from edge list to NEWICK format with language names
    """
    tree_list = []
    
    x, y = nodes_dict[node]
    
    if x not in leaves:
        tree_list.append(adjlist2newick(nodes_dict, x, leaves))
    else:
        tree_list.append(lang_dict[x])
    if y not in leaves:
        tree_list.append(adjlist2newick(nodes_dict, y, leaves))
    else:
        tree_list.append(lang_dict[y])
    print(tree_list)
    return "("+", ".join(map(str, tree_list))+")"

def adjlist2nodes_dict(edges_list):
    """Converts a adjacency list representation to a nodes dictionary
    which stores the information about neighboring nodes.
    """
    nodes_dict = defaultdict(list)
    for edge in edges_list:
        nodes_dict[edge[0]].append(edge[1])
    return nodes_dict

def adjlist2reverse_nodes_dict(edges_list):
    """Converts a adjacency list representation to a nodes dictionary
    which stores the information about neighboring nodes.
    """
    reverse_nodes_dict = defaultdict()
    for edge in edges_list:
        reverse_nodes_dict[edge[1]] = edge[0]
    return reverse_nodes_dict

def get_children(edges_list, leaves):
    """Get path from root to a tip"""
    root_node = 2*len(leaves) -1
    root2paths = defaultdict()
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

def get_subtree(edges_list, leaves):
    """Get leaves under a specific node"""
    children =get_children(edges_list, leaves)
    subtree_dict = defaultdict(list)
    
    for k, v in children.items():
        for l in v:
            subtree_dict[l].append(k)
    return subtree_dict

def swap_local_leaves(edges_list, leaves):
    """Swap two leaves within a subtree"""
    rev_nodes_dict = adjlist2reverse_nodes_dict(edges_list)
    hastings_ratio = 0.0
    subtree_dict = get_subtree(edges_list, leaves)
    root_node = 2*len(leaves) - 1
    internal_nodes = list(subtree_dict.keys())
    
    while(1):
        internal_node = random.choice(internal_nodes)
        if internal_node == root_node:
            continue
        else:
            subtree_leaves = subtree_dict[internal_node]
            x, y = random.sample(subtree_leaves, 2)
            #print("Swap leaves")
            #print("Random leaves ",x, y)
            rx, ry = rev_nodes_dict[x], rev_nodes_dict[y]
            del edges_list[rx, x]
            del edges_list[ry, y]
            edges_list[rx, y] = 1
            edges_list[ry, x] = 1
            break
    
    nodes_dict = adjlist2nodes_dict(edges_list)
    return edges_list, nodes_dict, hastings_ratio


def pairwise_distance(edges_list, leaves):
    paths = get_children(edges_list, leaves)
    pairwise = defaultdict(float)
    for leaf1, leaf2 in it.combinations_with_replacement(leaves, r=2):
        path1 = set(paths[leaf1])
        path2 = set(paths[leaf2])
        pairwise[leaf1,leaf2] = (len(path1)+len(path2)-(2.0*len(path1.intersection(path2))))#/max(len(path1), len(path2))
        pairwise[leaf2,leaf1] = pairwise[leaf1,leaf2]
    return pairwise

def shared_distance(edges_list, leaves):
    paths = get_children(edges_list, leaves)
    pairwise = defaultdict(float)
    for leaf1, leaf2 in it.combinations_with_replacement(leaves, r=2):
        path1 = set(paths[leaf1])
        path2 = set(paths[leaf2])
        pairwise[leaf1,leaf2] = float(len(path1.intersection(path2)))/max(len(path1), len(path2))
        pairwise[leaf2,leaf1] = pairwise[leaf1,leaf2]
    return pairwise



def swap_leaves(edges_list, leaves):
    """Swap two leaves randomly"""
    rev_nodes_dict = adjlist2reverse_nodes_dict(edges_list)
    hastings_ratio = 0.0
    while(1):
        x, y = random.sample(leaves, 2)
        #print("Swap leaves")
        #print("Random leaves ",x, y)
        rx, ry = rev_nodes_dict[x], rev_nodes_dict[y]
        if rx != ry:
            #print(edges_list)
            del edges_list[rx, x]
            del edges_list[ry, y]
            edges_list[rx, y] = 1
            edges_list[ry, x] = 1
            break
        else:
            continue
    
    nodes_dict = adjlist2nodes_dict(edges_list)
    return edges_list, nodes_dict, hastings_ratio

def externalSPR(edges_list, root_node, leaves):
    """Performs Subtree-Pruning and Regrafting
    Have to implement.
    """
    hastings_ratio = 0.0
    
    rev_nodes_dict = adjlist2reverse_nodes_dict(edges_list)
    nodes_dict = adjlist2nodes_dict(edges_list)
    
    #print("\n##### Old dictionary ########\n",nodes_dict,"\n")
    
    leaf = random.choice(leaves)
    parent_leaf = rev_nodes_dict[leaf]
    list_edges = list(edges_list.keys())
    #print("Selected random leaf, parent ", leaf, parent_leaf)

    tgt = random.choice(list_edges)
    #print("Target ",tgt)
    if parent_leaf == root_node or parent_leaf in tgt:
        hastings_ratio = 0.0
        #print("Rejected extSPR")
    elif rev_nodes_dict[parent_leaf] in tgt:
        hastings_ratio = 0.0
        #print("Rejected extSPR")
    else:
        del edges_list[rev_nodes_dict[parent_leaf], parent_leaf]
        children_parent_leaf = nodes_dict[parent_leaf]
        other_child_parent_leaf = children_parent_leaf[0]
        if leaf == other_child_parent_leaf:
            other_child_parent_leaf = children_parent_leaf[1]
        del edges_list[parent_leaf, other_child_parent_leaf]
        edges_list[tgt[0],parent_leaf] = 1
        edges_list[parent_leaf,tgt[1]] = 1
        edges_list[rev_nodes_dict[parent_leaf], other_child_parent_leaf]=1
        del edges_list[tgt]
        #print("Accepted extSPR")
    nodes_dict = adjlist2nodes_dict(edges_list)
    #print(nodes_dict,"\n")
    return edges_list, nodes_dict, hastings_ratio

def swap_top_children(edges_list, root_node, leaves):
    """Swaps two leaves within a subtree
    """
    nodes_dict = adjlist2nodes_dict(edges_list)
    left, right = nodes_dict[root_node]
    left_children = nodes_dict[left]
    right_children = nodes_dict[right]
    
    del edges_list[left, left_children[0]], edges_list[right, right_children[0]]
    edges_list[left,right_children[0]]=1
    edges_list[right,left_children[0]]=1

    nodes_dict = adjlist2nodes_dict(edges_list)
    temp_nodes_dict = adjlist2nodes_dict(edges_list)
    new_postorder = postorder(temp_nodes_dict, root_node, leaves)
    
    return edges_list, new_postorder

def scale_edge(temp_edges_dict):
    rand_edge = random.choice(list(temp_edges_dict))
    #rand_edge = next(iter(temp_edges_dict))
    rand_bl = temp_edges_dict[rand_edge]

    log_c = scaler_alpha*(np.random.uniform(0,1)-0.5)
    c = np.exp(log_c)
    rand_bl_new = rand_bl*c
    temp_edges_dict[rand_edge] = rand_bl_new

    #prior_ratio = expon.logpdf(rand_bl_new, scale=bl_exp_scale) - expon.logpdf(rand_bl, scale=bl_exp_scale)
    prior_ratio = bl_exp_scale*(rand_bl-rand_bl_new)
    
    return temp_edges_dict, log_c+prior_ratio

def rooted_NNI(temp_edges_list, root_node, leaves):
    """Performs Nearest Neighbor Interchange on a edges list.
    """
    #print("Selected NNI")
    hastings_ratio = 0.0
    #temp_edges_list = edges_list.copy()
    nodes_dict = adjlist2nodes_dict(temp_edges_list)

    #list_edges = list(temp_edges_list.keys())

    #random.shuffle(list_edges)
    rand_edge = None
    
    for x in temp_edges_list.keys():
        if x[0] not in leaves and x[1] not in leaves and x[0] != root_node:
            rand_edge = x
            break
    
    a, b = rand_edge

    x, y = nodes_dict[a], nodes_dict[b]
    
    #print(a,b,"\n")
    #print(x,y,"\n")
    
    if x[0] == b: tgt = x[1]
    else: tgt = x[0]

    #print("Target ",tgt)
    #random.shuffle(y)
    
    src_bl, tgt_bl = temp_edges_list[a,tgt], temp_edges_list[b,y[0]]
    del temp_edges_list[a,tgt], temp_edges_list[b,y[0]]
    temp_edges_list[a,y[0]] = tgt_bl
    temp_edges_list[b,tgt] = src_bl
    
    temp_nodes_dict = adjlist2nodes_dict(temp_edges_list)
    new_postorder = postorder(temp_nodes_dict, root_node, leaves)
    
    return temp_edges_list, new_postorder

