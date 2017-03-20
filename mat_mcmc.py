from collections import defaultdict
import subst_models, utils, params_moves, tree_helper
import sys, random
import numpy as np
from scipy import linalg
import multiprocessing as mp
np.random.seed(1234)
from scipy.stats import dirichlet, expon
import argparse
from functools import partial

n_chars, n_taxa, alphabet, taxa, n_sites = None, None, None, None, None
bl_exp_scale = 0.1
gemv = linalg.get_blas_funcs("gemv")

def site2mat(site):
    ll_mat = defaultdict(lambda: 1)
    zero_vec = np.zeros(n_chars)
    
    for k, v in site.items():
        if v in ["?", "-"]:
            x = np.ones(n_chars)
        else:
            x = np.zeros(n_chars)
            idx = alphabet.index(v)
            x[idx] = 1.0
        ll_mat[k] = x
    return ll_mat

def sites2Mat(sites):
    ll_mat = defaultdict(list)
    for k, v in sites.items():
        for ch in v:
            if ch in ["?", "-"]:
                x = np.ones(n_chars)
            else:
                x = np.zeros(n_chars)
                idx = alphabet.index(ch)
                x[idx] = 1.0
            ll_mat[k].append(x)

    for k, v in ll_mat.items():
        ll_mat[k] = np.array(v).T
        #print(k, np.array(v))
    return ll_mat


def update_LL(state):
    logLikehood = 0.0
    for idx in range(n_sites):
        logLikehood += ML(state["pi"], state["transitionMat"], state["postorder"], state["root"], ll_mats[idx])
    return logLikehood

def prior_probs(param, val):
    if param == "pi":
        return dirichlet.logpdf(val, alpha=prior_pi)
    elif param == "rates":
        return dirichlet.logpdf(val, alpha=prior_er)

def ML(pi, p_t, edges, root, ll_mat):
    #LL_mat = defaultdict(lambda: 1)
    LL_mat = ll_mat.copy()
    for parent, child in edges[::-1]:
        LL_mat[parent] *= np.dot(p_t[parent,child],LL_mat[child])
        
    return np.log(np.dot(LL_mat[root], pi))

def matML(state):
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
    

def get_prob_t(pi, rates, edges_dict, edges):
    p_t = defaultdict()
    for parent, child in edges[::-1]:
        if args.model == "F81":
            p_t[parent,child] = subst_models.ptF81(pi, edges_dict[parent,child])
        elif args.model == "JC":
            p_t[parent,child] =  subst_models.ptJC(n_chars, edges_dict[parent,child])
        elif args.model == "GTR":
            Q = subst_models.fnGTR(rates, pi)
            p_t[parent,child] = linalg.expm2(Q*edges_dict[parent,child])
    return p_t
    

def initialize():
    state = defaultdict()
    pi, er = subst_models.init_pi_er(n_chars, args.model)
    state["pi"] = pi
    state["rates"] = er
    state["tree"], state["root"] = tree_helper.init_tree(taxa)
    nodes_dict = tree_helper.adjlist2nodes_dict(state["tree"])
    edges_ordered_list = tree_helper.postorder(nodes_dict, state["root"], taxa)
    state["postorder"] = edges_ordered_list
    state["transitionMat"] = get_prob_t(state["pi"], state["rates"], state["tree"], state["postorder"])
    return state

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Input a file in Phylip format with taxa and characters separated by a TAB character",  type=str)
parser.add_argument("-m", "--model", help="JC/F81/GTR",  type=str)
parser.add_argument("-n","--n_gen", help="Number of generations",  type=int)
parser.add_argument("-t","--thin", help="Number of generations",  type=int)
args = parser.parse_args()

n_taxa, n_chars, alphabet, site_dict, taxa, n_sites = utils.readPhy(args.input_file)
n_rates = int(n_chars*(n_chars-1)/2)
prior_pi = np.array([1]*n_chars)
prior_er = np.array([1]*n_rates)
iu1 = np.triu_indices(n_chars,1)


#sites = utils.transform(site_dict)
#ll_mats = [site2mat(sites[idx]) for idx in range(n_sites)]

ll_mats = sites2Mat(site_dict)


init_state = initialize()
init_state["logLikehood"] = matML(init_state)
state = init_state.copy()
print(init_state["logLikehood"])

#sys.exit(1)

if args.model == "F81":
    params_list = ["pi", "bl", "tree"]
    weights = [0.2, 0.5, 0.3]
elif args.model == "GTR":
    params_list = ["pi","rates", "tree", "bl"]#, "tree"]#tree", "bl"]
    weights = [0.1, 0.1, 0.4, 0.4]#0.4, 0.4, ]#, 0.4]
elif args.model == "JC":
    params_list = ["bl", "tree"]
    weights = [0.5, 0.5]


tree_move_weights = [0.6, 0.0, 0.4]
moves_count = defaultdict(float)
accepts_count = defaultdict(float)
moves_dict = {"pi": [params_moves.mvDirichlet], "rates": [params_moves.mvDualSlider], "tree":[tree_helper.rooted_NNI, tree_helper.NNI_swap_subtree,tree_helper.externalSPR], "bl":[tree_helper.scale_edge]}

n_accepts = 0.0
samples = []

params_fileWriter = open(args.input_file.split("/")[1]+".params","w")
print("Iteration", "logLikehood", "Tree Length", sep="\t", file=params_fileWriter)

for n_iter in range(1, args.n_gen+1):
    propose_state = state.copy()
    current_ll, proposed_ll, ll_ratio, hr = 0.0, 0.0, 0.0, 0.0
    
    param_select = np.random.choice(params_list, p=weights)
    if param_select == "tree":
        move = np.random.choice(moves_dict[param_select], p=tree_move_weights)
    else:
        move = np.random.choice(moves_dict[param_select])
    #print("Selected move ", param_select, move.__name__)

    moves_count[param_select,move.__name__] += 1
    if param_select in ["pi", "rates"]:
        new_param, hr = move(propose_state[param_select])
        propose_state[param_select] = new_param
        current_prior = prior_probs(param_select, state[param_select])
        prop_prior = prior_probs(param_select, propose_state[param_select])
        hr += prop_prior-current_prior
        
    elif param_select == "bl":
        temp_edges_dict, hr = move(propose_state["tree"].copy())
        propose_state["tree"] = temp_edges_dict
        
    elif param_select == "tree":
        temp_edges_dict, prop_post_order, hr = move(propose_state[param_select].copy(), propose_state["root"], taxa)
        propose_state["tree"] = temp_edges_dict
        propose_state["postorder"] = prop_post_order
    
    propose_state["transitionMat"] = get_prob_t(propose_state["pi"], propose_state["rates"], propose_state["tree"], propose_state["postorder"])
    
    current_ll = state["logLikehood"]
    proposed_ll = matML(propose_state)

    ll_ratio = proposed_ll - current_ll + hr

    if np.log(random.random()) < ll_ratio:
        n_accepts += 1
        if param_select == "bl":
            state["tree"] = propose_state["tree"]
        elif param_select == "tree":
            state[param_select] = propose_state[param_select]
            state["postorder"] = propose_state["postorder"]
        else:
            state[param_select] = propose_state[param_select]
        state["transitionMat"] = propose_state["transitionMat"].copy()
        state["logLikehood"] = proposed_ll
        accepts_count[param_select,move.__name__] += 1
        #if n_iter % args.thin == 0:
        #print(n_accepts, n_iter, state["logLikehood"])
        #   print(state["tree"])
    #del propose_state
    if n_iter % args.thin == 0:
        TL = sum(state["tree"].values())
        sampled_tree = tree_helper.adjlist2newickBL(state["tree"], tree_helper.adjlist2nodes_dict(state["tree"]), state["root"], taxa)+";"
        print(n_iter, state["logLikehood"], TL)
        print(n_iter, state["logLikehood"], TL, sep="\t", file=params_fileWriter)
        #print(sampled_tree)


for k, v in moves_count.items():
    print(k, accepts_count[k], v)


