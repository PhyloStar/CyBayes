#from __future__ import print_function
import argparse, utils
from mcmc_gamma import *
from ML_gamma import *
import config
from collections import defaultdict
np.random.seed(1234)
random.seed(1234)
#global N_TAXA, N_CHARS, ALPHABET, LEAF_LLMAT, TAXA, MODEL, IN_DTYPE, NORM_BETA

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Input a file in Phylip format with taxa and characters separated by a TAB character",  type=str)
parser.add_argument("-m", "--model", help="JC/F81/GTR",  type=str)
parser.add_argument("-n","--n_gen", help="Number of generations",  type=int)
parser.add_argument("-t","--thin", help="Number of generations after to print to file",  type=int)
parser.add_argument("-d","--data_type", help="Type of data if it is binary/multistate. Multistate characters should be separated by a space whereas binary need not be. Specify bin for binary and multi for multistate characters or phonetic alignments",  type=str)
parser.add_argument("-o","--output_file", help="Name of the out file prefix",  type=str)
args = parser.parse_args()

if args.data_type == "bin":
    config.N_TAXA, config.N_CHARS, config.ALPHABET, site_dict, config.LEAF_LLMAT, config.TAXA, config.N_SITES = utils.readBinaryPhy(args.input_file)
    print(config.N_TAXA)
elif args.data_type == "multi":
    config.N_TAXA, config.N_CHARS, config.ALPHABET, site_dict, config.LEAF_LLMAT, config.TAXA, config.N_SITES = utils.readMultiPhy(args.input_file)


config.IN_DTYPE = args.data_type
config.N_GEN = args.n_gen
config.THIN = args.thin
config.MODEL = args.model
config.N_NODES = 2*config.N_TAXA -1

print("Characters ", config.N_CHARS)
print("TAXA ", config.TAXA)
print("Number of TAXA ", config.N_TAXA)
print("Alphabet ", config.ALPHABET)


n_rates = config.N_CHARS*(config.N_CHARS-1)//2
prior_pi = np.array([1]*config.N_CHARS)
prior_er = np.array([1]*n_rates)

if config.MODEL == "JC":
    config.NORM_BETA = config.N_CHARS/(config.N_CHARS-1)

init_state = state_init()
print(init_state["pi"], init_state["srates"])

site_rates = get_siterates(init_state["srates"])
print(site_rates)

cache_LL_Mat, cache_paths_dict = None, None
init_state["logLikehood"], cache_LL_mats = matML(init_state, config.TAXA, config.LEAF_LLMAT)

state = init_state.copy()
init_tree = adjlist2newickBL(state["tree"], adjlist2nodes_dict(state["tree"]), state["root"])+";"
cache_paths_dict = adjlist2reverse_nodes_dict(state["tree"])

print("Initial Random Tree ", init_tree, sep="\t")

print("Initial Likelihood ",init_state["logLikehood"])

if config.MODEL == "F81":
    params_list = ["pi", "bl", "tree", "srates"]
    weights = np.array([1, 5, 5, 1], dtype=np.float64)
elif config.MODEL == "GTR":
    params_list = ["pi","rates", "tree", "bl", "srates"]
    weights = np.array([1, 2, 5, 20,1], dtype=np.float64)
elif config.MODEL == "JC":
    params_list = ["bl", "tree", "srates"]
    weights = np.array([5, 5, 1], dtype=np.float64)

tree_move_weights = np.array([5,5], dtype=np.float64)
bl_move_weights = np.array([10], dtype=np.float64)

weights = weights/np.sum(weights)
tree_move_weights = tree_move_weights/np.sum(tree_move_weights)
bl_move_weights = bl_move_weights/np.sum(bl_move_weights)

moves_count = defaultdict(int)
accepts_count = defaultdict(int)
moves_dict = {"pi": [mvDualSlider], "rates": [mvDualSlider], "tree":[rooted_NNI, externalSPR], "bl":[scale_edge], "srates":[scale_alpha]}
n_accepts = 0.0

params_fileWriter = open(args.output_file+".params","w")
trees_fileWriter = open(args.output_file+".trees","w")
const_states = ["pi("+idx+")" for idx in config.ALPHABET]



#print("Iter", "LnL", "TL", *const_states, sep="\t", file=params_fileWriter)
print("Iter", "LnL", "TL", "Alpha", sep="\t", file=params_fileWriter)
#print "Iteration", "logLikehood", "Tree Length",const_states

for n_iter in range(1,  config.N_GEN+1):
    propose_state = state.copy()
    
    current_ll, proposed_ll, ll_ratio, hr, change_edge, pr_ratio = 0.0, 0.0, 0.0, 0.0, None, 0.0
        
    param_select = np.random.choice(params_list, p=weights)
        
    current_rates = site_rates[:]

    if param_select == "tree":
        move = np.random.choice(moves_dict[param_select], p=tree_move_weights)
    elif param_select == "bl":
        move = np.random.choice(moves_dict[param_select], p=bl_move_weights)
    else:
        move = np.random.choice(moves_dict[param_select])
    
    moves_count[param_select,move.__name__] += 1
    
    if param_select in ["pi", "rates"]:
        new_param, hr = move(state[param_select].copy())
        propose_state[param_select] = new_param
    elif param_select == "bl":
        prop_edges_dict, hr, pr_ratio, change_edge = move(state["tree"].copy())
        propose_state["tree"] = prop_edges_dict
    elif param_select == "tree":
        if move.__name__ == "rooted_NNI":
            prop_edges_dict, prop_post_order, hr, nodes_recompute = move(state[param_select].copy(), state["root"])
            #print(src1, src2, tgt1, tgt2)
        else:
            prop_edges_dict, prop_post_order, hr = move(state[param_select].copy(), state["root"])
        #prop_edges_dict, prop_post_order, hr = move(state[param_select].copy(), state["root"], taxa)
        propose_state["tree"] = prop_edges_dict
        propose_state["postorder"] = prop_post_order
    elif param_select == "srates":
        new_param, hr, pr_ratio = move(state[param_select])
        propose_state[param_select] = new_param
        site_rates = get_siterates(new_param)

    #if move.__name__ == "scale_edge":
        #propose_state["transitionMat"] = get_copy_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"], state["transitionMat"], change_edge)
    #    old_edge_p_t = state["transitionMat"][change_edge]
    #    propose_state["transitionMat"] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_edge], state["transitionMat"], change_edge)
        #print("Change edge ", change_edge[1], state["root"])
    #    nodes_recompute = get_path2root(cache_paths_dict, change_edge[1], state["root"])
    #    proposed_ll, proposed_llMat = cache_matML(propose_state, config.TAXA, config.LEAF_LLMAT, cache_LL_Mat, nodes_recompute)
    #elif move.__name__ == "rooted_NNI":
#            nodes_recompute = [src2[0]]+tree_helper.get_path2root(tree_helper.adjlist2reverse_nodes_dict(prop_edges_dict), src2[0], state["root"])
    #    propose_state["transitionMat"] = get_prob_t(propose_state["pi"], propose_state["tree"], propose_state["rates"])
    #    proposed_ll, proposed_llMat = cache_matML(propose_state, config.TAXA, config.LEAF_LLMAT, cache_LL_Mat, nodes_recompute)

    propose_state["transitionMat"] = [get_prob_t(propose_state["pi"], propose_state["tree"], propose_state["rates"], mean_rate) for mean_rate in site_rates]
    proposed_ll, proposed_llMat = matML(propose_state, config.TAXA, config.LEAF_LLMAT)

    current_ll = state["logLikehood"]
    ll_ratio = proposed_ll - current_ll + pr_ratio
    ll_ratio += hr
    
    if np.log(random.random()) < ll_ratio:
        n_accepts += 1
        if param_select == "bl":
            state["tree"] = propose_state["tree"]
        elif param_select == "tree":
            state[param_select] = propose_state[param_select]
            state["postorder"] = prop_post_order
            cache_paths_dict = adjlist2reverse_nodes_dict(state[param_select])
            cache_LL_Mat = proposed_llMat
        else:
            state[param_select] = propose_state[param_select]
            
        state["transitionMat"] = propose_state["transitionMat"]
        state["logLikehood"] = proposed_ll
        
        accepts_count[param_select,move.__name__] += 1
    else:
        if param_select == "srates":
            site_rates = current_rates[:]
    #    if move.__name__ == "scale_edge":
    #        state["transitionMat"][change_edge] = old_edge_p_t
    
        #TL = sum(state["tree"].values())
        #print(n_iter, state["logLikehood"], proposed_ll, current_ll,TL, param_select, move.__name__, sep="\t", flush=True)

    #del propose_state
    if n_iter % config.THIN == 0:
        TL = sum(state["tree"].values())
        stationary_freqs = "\t".join([str(state["pi"][idx]) for idx in range(config.N_CHARS)])
        sampled_tree = adjlist2newickBL(state["tree"], adjlist2nodes_dict(state["tree"]), state["root"])+";"
        print(n_iter, state["logLikehood"], proposed_ll, TL, param_select, move.__name__, sep="\t")
        print(n_iter, state["logLikehood"], TL, state["srates"], sep="\t", file=params_fileWriter)
        
        #print(n_iter, sampled_tree, state["logLikehood"], sep="\t", file=trees_fileWriter)
        print(n_iter, sampled_tree, sep="\t", file=trees_fileWriter)

#params_fileWriter.close()
#trees_fileWriter.close()
for k, v in moves_count.items():
    print(k, accepts_count[k], v)


