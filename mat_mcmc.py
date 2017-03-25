from collections import defaultdict
import subst_models, utils, params_moves, tree_helper
import sys, random, copy
import numpy as np
from scipy import linalg
import multiprocessing as mp
np.random.seed(1234)
from scipy.stats import dirichlet, expon
import argparse, math
from ML import matML_dot, matML_inplace, matML_inplace_bl

n_chars, n_taxa, alphabet, taxa, n_sites, model_normalizing_beta = None, None, None, None, None, 1.0

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

def prior_probs(param, val):
    if param == "pi":
        return dirichlet.logpdf(val, alpha=prior_pi)
    elif param == "rates":
        return dirichlet.logpdf(val, alpha=prior_er)

def get_copy_transition_mat(pi, rates, edges_dict, edges, transition_mat, change_edge):
    if args.model == "F81":
        model_normalizing_beta = 1/(1-np.dot(pi, pi))
    elif args.model == "JC":
        model_normalizing_beta = n_chars/(n_chars-1)
        
    new_transition_mat = defaultdict()
    
    for Edge in edges[::-1]:
        parent, child = Edge
        if Edge != change_edge:
            new_transition_mat[parent, child] = transition_mat[parent, child].copy()        
        else:
            d = edges_dict[parent,child]
            if args.model == "F81":
                if args.data_type == "multi":
                    new_transition_mat[parent,child] = subst_models.ptF81(pi, d)
                elif args.data_type == "bin":
                    x = math.exp(-model_normalizing_beta*d)
                    y = 1.0-x
                    new_transition_mat[parent,child] = subst_models.binaryptF81(pi, x, y)
            elif args.model == "JC":
                x = math.exp(-model_normalizing_beta*d)
                y = (1.0-x)/n_chars        
                #p_t[parent,child] =  subst_models.fastJC(n_chars, x, y)
                new_transition_mat[parent,child] =  subst_models.ptJC(n_chars, x, y)
            elif args.model == "GTR":
                Q = subst_models.fnGTR(rates, pi)
                new_transition_mat[parent,child] = linalg.expm2(Q*edges_dict[parent,child])
    return new_transition_mat


def get_prob_t(pi, rates, edges_dict, edges):
    p_t = defaultdict()

    if args.model == "F81":
        model_normalizing_beta = 1/(1-np.dot(pi, pi))
    elif args.model == "JC":
        model_normalizing_beta = n_chars/(n_chars-1)
        
    for parent, child in edges[::-1]:
        d = edges_dict[parent,child]
        if args.model == "F81":
            if args.data_type == "multi":
                p_t[parent,child] = subst_models.ptF81(pi, d)
            elif args.data_type == "bin":
                x = math.exp(-model_normalizing_beta*d)
                y = 1.0-x
                p_t[parent,child] = subst_models.binaryptF81(pi, x, y)
        elif args.model == "JC":
            x = math.exp(-model_normalizing_beta*d)
            y = (1.0-x)/n_chars        
            #p_t[parent,child] =  subst_models.fastJC(n_chars, x, y)
            p_t[parent,child] =  subst_models.ptJC(n_chars, x, y)
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
parser.add_argument("-t","--thin", help="Number of generations after to print to file",  type=int)
parser.add_argument("-d","--data_type", help="Type of data if it is binary/multistate. Multistate characters should be separated by a space whereas binary need not be. Specify bin for binary and multi for multistate characters or phonetic alignments",  type=str)
parser.add_argument("-o","--output_file", help="Name of the out file prefix",  type=str)
args = parser.parse_args()

if args.data_type == "bin":
    n_taxa, n_chars, alphabet, site_dict, ll_mats, taxa, n_sites = utils.readBinaryPhy(args.input_file)
elif args.data_type == "multi":
    n_taxa, n_chars, alphabet, site_dict, ll_mats, taxa, n_sites = utils.readPhy(args.input_file)

n_rates = int(n_chars*(n_chars-1)/2)
prior_pi = np.array([1]*n_chars)
prior_er = np.array([1]*n_rates)

print("Languages ", taxa)
print("Alphabet ", alphabet)

init_state = initialize()
#init_state["logLikehood"], init_state["logLikehoodMat"] = matML_inplace(init_state, taxa, ll_mats)

init_state["logLikehood"] = matML_inplace(init_state, taxa, ll_mats)

state = init_state.copy()
init_tree = tree_helper.adjlist2newickBL(state["tree"], tree_helper.adjlist2nodes_dict(state["tree"]), state["root"], taxa)+";"
print("Initial Random Tree ")
print(init_tree)
print("Likelihood ",init_state["logLikehood"])


if args.model == "JC":
    model_normalizing_beta = n_chars/(n_chars-1)

if args.model == "F81":
    params_list = ["pi", "bl", "tree"]
    weights = np.array([1, 20, 15])
elif args.model == "GTR":
    params_list = ["pi","rates", "tree", "bl"]
    weights = np.array([1, 2, 15, 20])
elif args.model == "JC":
    params_list = ["bl", "tree"]
    weights = np.array([20, 15])

tree_move_weights = np.array([5,0,5])
bl_move_weights = np.array([10])

weights = weights/sum(weights)
tree_move_weights = tree_move_weights/sum(tree_move_weights)
bl_move_weights = bl_move_weights/sum(bl_move_weights)

moves_count = defaultdict(float)
accepts_count = defaultdict(float)
moves_dict = {"pi": [params_moves.mvDualSlider], "rates": [params_moves.mvDualSlider], "tree":[tree_helper.rooted_NNI, tree_helper.NNI_swap_subtree,tree_helper.externalSPR], "bl":[tree_helper.scale_edge]}#, tree_helper.scale_all_edges]}

n_accepts = 0.0
samples = []

params_fileWriter = open(args.output_file+".params","w")
trees_fileWriter = open(args.output_file+".trees","w")

const_states = "\t".join(["pi"+idx for idx in alphabet])

print("Iteration", "logLikehood", "Tree Length",const_states, sep="\t", file=params_fileWriter)

for n_iter in range(1, args.n_gen+1):
    propose_state = state.copy()
    current_ll, proposed_ll, ll_ratio, hr = 0.0, 0.0, 0.0, 0.0
    
    param_select = np.random.choice(params_list, p=weights)
    if param_select == "tree":
        move = np.random.choice(moves_dict[param_select], p=tree_move_weights)
    elif param_select == "bl":
        move = np.random.choice(moves_dict[param_select], p=bl_move_weights)
    else:
        move = np.random.choice(moves_dict[param_select])
    #print("Selected move ", param_select, move.__name__)

    change_edge = None
    
    moves_count[param_select,move.__name__] += 1
    
    if param_select in ["pi", "rates"]:
        new_param, hr = move(propose_state[param_select].copy())
        propose_state[param_select] = new_param
        
    elif param_select == "bl":
        temp_edges_dict, hr, change_edge = move(propose_state["tree"].copy())
        propose_state["tree"] = temp_edges_dict
        
    elif param_select == "tree":
        temp_edges_dict, prop_post_order, hr = move(propose_state[param_select].copy(), propose_state["root"], taxa)
        propose_state["tree"] = temp_edges_dict
        propose_state["postorder"] = prop_post_order
    
    if move.__name__ == "scale_edge":    
        propose_state["transitionMat"] = get_copy_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"], propose_state["postorder"], state["transitionMat"], change_edge)
    else:
        propose_state["transitionMat"] = get_prob_t(propose_state["pi"], propose_state["rates"], propose_state["tree"], propose_state["postorder"])
    
    current_ll = state["logLikehood"]
    
    #if move.__name__ == "scale_edge":
    #    proposed_ll, proposed_llMat = matML_inplace_bl(state, taxa, ll_mats, state["logLikehoodMat"], change_edge)
    #else:
    #    proposed_ll, proposed_llMat = matML_inplace(propose_state, taxa, ll_mats)
        
    #proposed_ll, proposed_llMat = matML_inplace(propose_state, taxa, ll_mats)
    proposed_ll = matML_inplace(propose_state, taxa, ll_mats)
        
    ll_ratio = proposed_ll - current_ll + hr

    if math.log(random.random()) < ll_ratio:
        n_accepts += 1
        if param_select == "bl":
            state["tree"] = propose_state["tree"]
        elif param_select == "tree":
            state[param_select] = propose_state[param_select]
            state["postorder"] = propose_state["postorder"]
        else:
            state[param_select] = propose_state[param_select]
            
        state["transitionMat"] = propose_state["transitionMat"]
        state["logLikehood"] = proposed_ll
        #state["logLikehoodMat"] = proposed_llMat
        accepts_count[param_select,move.__name__] += 1
        #TL = sum(state["tree"].values())
        #print(n_iter, state["logLikehood"], proposed_ll, current_ll,TL, state["pi"], param_select, move.__name__, sep="\t", flush=True)

    #del propose_state
    if n_iter % args.thin == 0:
        TL = sum(state["tree"].values())
        stationary_freqs = "\t".join([str(state["pi"][idx]) for idx in range(n_chars)])
        sampled_tree = tree_helper.adjlist2newickBL(state["tree"], tree_helper.adjlist2nodes_dict(state["tree"]), state["root"], taxa)+";"
        print(n_iter, state["logLikehood"], proposed_ll, current_ll, TL, state["pi"], param_select, move.__name__, sep="\t")
        print(n_iter, state["logLikehood"], TL, stationary_freqs, sep="\t", file=params_fileWriter, flush=True)
        print(n_iter, sampled_tree, state["logLikehood"], sep="\t", file=trees_fileWriter)

params_fileWriter.close()
trees_fileWriter.close()
for k, v in moves_count.items():
    print(k, accepts_count[k], v)


