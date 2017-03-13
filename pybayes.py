from collections import defaultdict
import subst_models, utils, params_moves, tree_helper
import sys
import numpy as np
from scipy import linalg
import multiprocessing as mp
np.random.seed(1234)

n_chars, n_taxa, alphabet, taxa, n_sites = None, None, None, None, None

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

def ML(mcmc_state, ll_mat):
    LL_mat = defaultdict(lambda: 1)
    
    edges_dict = mcmc_state["tree"]
    edges = mcmc_state["postorder"]
    p_t = mcmc_state["transitionMat"]
    
    
    for parent, child in edges[::-1]:
        if child in taxa:
            LL_mat[parent] = LL_mat[parent]*np.dot(p_t[parent,child],ll_mat[child])
        else:
            LL_mat[parent] = LL_mat[parent]*np.dot(p_t[parent,child],LL_mat[child])
    
    #print("LL_MAT at root \n", ll_mat[mcmc_state["root"]],"\n")
    #print(mcmc_state["pi"],"\n")
    #ll = 
    #print(ll)
    return np.log(np.dot(LL_mat[mcmc_state["root"]],mcmc_state["pi"]))

def get_prob_t(mcmc_state):
    p_t = defaultdict()
    edges_dict = mcmc_state["tree"]
    edges = mcmc_state["postorder"]
    for parent, child in edges[::-1]:
        if model == "F81":
            p_t[parent,child] = subst_models.ptF81(mcmc_state["pi"], edges_dict[parent,child])
        elif model == "JC":
            p_t[parent,child] =  subst_models.ptJC(mcmc_state["pi"], edges_dict[parent,child])
        elif model == "GTR":
            Q = subst_models.fnGTR(mcmc_state["rates"], mcmc_state["pi"])
            p_t[parent,child] = linalg.expm(Q*edges_dict[parent,child])
    return p_t

def initialize():
    mcmc_state = defaultdict()
    pi, er = subst_models.init_pi_er(n_chars, model)
    mcmc_state["pi"] = pi
    mcmc_state["rates"] = er
    mcmc_state["tree"], mcmc_state["root"] = tree_helper.init_tree(taxa)
    nodes_dict = tree_helper.adjlist2nodes_dict(mcmc_state["tree"])
    edges_ordered_list = tree_helper.postorder(nodes_dict, mcmc_state["root"], taxa)
    mcmc_state["postorder"] = edges_ordered_list
    mcmc_state["transitionMat"] = get_prob_t(mcmc_state)
    return mcmc_state


input_file = sys.argv[1]
model = sys.argv[2]
n_generations = sys.argv[3]

n_taxa, n_chars, alphabet, site_dict, taxa, n_sites = utils.readPhy(input_file)
#print("ALPHABET ", n_chars)
sites = utils.transform(site_dict)

mcmc_state = initialize()
#mcmc_states = [ for idx in range(n_sites)]

utils.print_mcmcstate(mcmc_state)
print("\n")

#pool = mp.Pool(2)
#tree_ll = sum(pool.map())

ll_mats = [site2mat(sites[idx]) for idx in range(n_sites)]

for n_iter in range(10000):
    tree_ll = 0.0
    mcmc_state = initialize()
    for idx in range(n_sites):
        tree_ll += ML(mcmc_state, ll_mats[idx])

    print("Total likelihood ", n_iter, tree_ll)











