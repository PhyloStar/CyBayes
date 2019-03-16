#Start the MCMC analysis with MAP tree

#from __future__ import print_function
import argparse, utils
from mcmc_gamma import *
#from ML_gamma import *
#from PyML import *
from CyML import *
import config
from collections import defaultdict
import itertools as it
import multiprocessing as mp
np.random.seed(1234)
random.seed(1234)
from functools import partial

#global N_TAXA, N_CHARS, ALPHABET, LEAF_LLMAT, TAXA, MODEL, IN_DTYPE, NORM_BETA

def get_path(d, internal_node, root):
    paths = []
    while(1):
        parent = d[internal_node]
        #paths += [parent]
        paths.append(parent)
        internal_node = parent
        if parent == root:
            break

    return paths

def get_paths(edges, root):
    paths_dict = defaultdict(list)
    d = {v:k for k,v in edges}
    for i in range(1, config.N_TAXA+1):
        paths_dict[i] = get_path(d, i, root)
#        print(i, paths_dict[i])
    return paths_dict

#d = {v:k for k,v in edges}
#paths_dict = get_paths(edges, root)

def get_mrca(paths_dict, taxa, root):
    mrca = None
    
    for i in taxa:
        if mrca is None:
            mrca = paths_dict[i][::-1]
        else:
            temp_mrca = []
            for x, y in zip(mrca, paths_dict[i][::-1]):
                if x == y: temp_mrca.append(x)
                else: break
#            print(i, mrca, paths_dict[i], temp_mrca)
            mrca = temp_mrca[:]
    return mrca[-1]

def get_mrca_list(edges, cogset_taxa_list, root):
#    print(edges)
    paths_dict = get_paths(edges, root)
    mrca_list = []
    for taxa in cogset_taxa_list:
        mrca_list.append(get_mrca(paths_dict, taxa, root))
#        print(taxa, mrca_list[-1])
    return mrca_list

T_i, mult_scale, T_stop = 20, 0.95, 10e-5 # Log scheduling does not work so well
threshold = 0.0001

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Input a file in Phylip format with taxa and characters separated by a TAB character",  type=str)
parser.add_argument("-m", "--model", help="JC/F81/GTR",  type=str)
parser.add_argument("-n","--n_gen", help="Number of generations",  type=int)
parser.add_argument("-t","--thin", help="Number of generations after to print to file",  type=int)
parser.add_argument("-d","--data_type", help="Type of data if it is binary/multistate. Multistate characters should be separated by a space whereas binary need not be. Specify bin for binary and multi for multistate characters or phonetic alignments. It could also be cognate alignments separated by newlines",  type=str)
parser.add_argument("-o","--output_file", help="Name of the out file prefix",  type=str)
parser.add_argument("-pi","--base_freqs", help="Fixed Base Frequencies",  type=str)
parser.add_argument("-T","--init_temp", help="Initial Temperature",  type=int)
args = parser.parse_args()

if args.data_type == "bin":
    config.N_TAXA, config.N_CHARS, config.ALPHABET, site_dict, config.LEAF_LLMAT, config.TAXA, config.N_SITES, alphabet_counter = utils.readBinaryPhy(args.input_file)

elif args.data_type == "multi":
    config.N_TAXA, config.N_CHARS, config.ALPHABET, site_dict, config.LEAF_LLMAT, config.TAXA, config.N_SITES = utils.readMultiPhy(args.input_file)

elif args.data_type == "cognate":
    config.N_TAXA, config.N_CHARS, config.ALPHABET, config.LEAF_LLMAT_LIST, config.TAXA, cogset_taxa_list, alphabet_counter = utils.readCognatePhy(args.input_file)

if args.data_type == "cognate":
    config.IN_DTYPE = "multi"
else:
    config.IN_DTYPE = args.data_type

config.N_GEN = args.n_gen
config.THIN = args.thin
config.MODEL = args.model
config.N_NODES = 2*config.N_TAXA -1
config.N_COG_CLASSES = len(config.LEAF_LLMAT_LIST)
T_i = args.init_temp
step_temp = 2*config.N_TAXA

print("Characters ", config.N_CHARS)
print("TAXA ", config.TAXA)
print("Number of TAXA ", config.N_TAXA)
print("Alphabet ", config.ALPHABET)
print("Categories ", config.N_CATS)

n_rates = config.N_CHARS*(config.N_CHARS-1)//2
prior_pi = np.array([1]*config.N_CHARS)
prior_er = np.array([1]*n_rates)

if config.MODEL == "JC":
    config.NORM_BETA = config.N_CHARS/(config.N_CHARS-1)

init_state = state_init()

#print("Init state PI ",np.asarray(init_state["pi"]).shape)
#print(np.asarray(init_state["rates"]).shape)

if config.N_CATS > 1:
    site_rates = get_siterates(init_state["srates"])
else:
    site_rates = init_state["srates"]

config.N_BRANCHES = len(init_state["postorder"])

cache_LL_Mats, cache_paths_dict = None, None

mrca_list = get_mrca_list(init_state["postorder"], cogset_taxa_list, init_state["root"]) #obtain the mrca_list

print("Number of cognate classes = {}".format(len(config.LEAF_LLMAT_LIST)), sep="\n")

if args.base_freqs == "ml":
    norm_sum = sum(list(alphabet_counter.values()))
    base_freqs = [alphabet_counter[x]/norm_sum for x in config.ALPHABET]
    init_state["pi"] = np.array(base_freqs)

init_state["logLikehood"], cache_LL_Mats = cognateMatML2(init_state["pi"], init_state["root"], config.LEAF_LLMAT_LIST, init_state["postorder"], init_state["transitionMat"], config.N_CATS, mrca_list, config.N_COG_CLASSES)

leaves_mat_list = []

for llmat in config.LEAF_LLMAT_LIST:
    lmat = {}
    for k in llmat:
        lmat[k] = np.asarray(llmat[k])
    leaves_mat_list.append(lmat)

#print("PI ", type(init_state["pi"]))
#print("PI ", np.asarray(init_state["pi"]))
#print("Root ", type(init_state["root"]))
#print("postorder ", type(init_state["postorder"]))
#print(type(config.LEAF_LLMAT_LIST[0][1]))
#print(type(init_state["transitionMat"][0][40,17]))

#init_state["logLikehood"] = parCognateMatML(init_state["pi"], init_state["root"], leaves_mat_list, init_state["postorder"], init_state["transitionMat"], config.N_CATS, mrca_list)


#print(cache_LL_Mats)

state = init_state.copy()
init_tree = adjlist2newickBL(state["tree"], adjlist2nodes_dict(state["tree"]), state["root"])+";"
cache_paths_dict = adjlist2reverse_nodes_dict(state["tree"]) #Caches the paths dictionary

print("Initial Random Tree ", init_tree, sep="\t")

print("Initial Likelihood ",init_state["logLikehood"])

if config.MODEL == "F81":
    params_list = ["pi", "tree", "bl", "srates"]
    weights = np.array([0.5, 3, 4, 0.5], dtype=np.float64)
elif config.MODEL == "GTR":
    params_list = ["pi","rates", "tree", "bl", "srates"]
    weights = np.array([0.5, 0.5, 3, 4, 0.5], dtype=np.float64)
elif config.MODEL == "JC":
    params_list = ["bl", "tree", "srates"]
    weights = np.array([4, 3, 0.5], dtype=np.float64)

if config.N_CATS == 1: weights[-1] = 0.0

#if args.base_freqs == "fixed": weights[0] = 0.0

tree_move_weights = np.array([4, 1], dtype=np.float64)
bl_move_weights = np.array([3, 1], dtype=np.float64)

weights = weights/np.sum(weights)
tree_move_weights = tree_move_weights/np.sum(tree_move_weights)
bl_move_weights = bl_move_weights/np.sum(bl_move_weights)


moves_count = defaultdict(int)
accepts_count = defaultdict(int)
moves_dict = {"pi": [mvDualSlider], "rates": [mvDualSlider], "tree":[rooted_NNI, externalSPR], "bl":[scale_edge, node_slider], "srates":[scale_alpha]}

params_fileWriter = open(args.output_file+".log","w")
trees_fileWriter = open(args.output_file+".trees","w")
const_states = ["pi("+idx+")" for idx in config.ALPHABET]

print("Iter", "LnL", "TL", "Alpha", sep="\t", file=params_fileWriter)

propose_state, prop_tmats = {}, []

#for n_iter in range(1,  config.N_GEN+1):

n_iter = 1

while(1):
    propose_state["pi"],  propose_state["rates"], propose_state["srates"], propose_state["tree"], propose_state["postorder"] = state["pi"].copy(), state["rates"].copy(), state["srates"], state["tree"].copy(), state["postorder"].copy()

    current_ll, proposed_ll, ll_ratio, hr, change_edge, pr_ratio, old_edge_p_ts, old_edge_p_t_as = 0.0, 0.0, 0.0, 0.0, None, 0.0, [], []
        
    param_select = np.random.choice(params_list, p=weights)

    if param_select == "tree":
        move = np.random.choice(moves_dict[param_select], p=tree_move_weights)
    elif param_select == "bl":
        move = np.random.choice(moves_dict[param_select], p=bl_move_weights)
    else:
        move = np.random.choice(moves_dict[param_select])
    
    moves_count[param_select,move.__name__] += 1

#    print("Move name ",move.__name__)
    
    if param_select in ["pi", "rates"]:
        new_param, hr = move(state[param_select].copy())
        propose_state[param_select] = new_param

    elif param_select == "bl":

        if move.__name__ == "scale_edge":
            prop_edges_dict, hr, pr_ratio, change_edge = move(state["tree"].copy())

        elif move.__name__ == "node_slider":
            prop_edges_dict, hr, pr_ratio, change_edge, change_parent_edge = move(state["tree"].copy(), state["root"])

        nodes_recompute = get_path2root(cache_paths_dict, change_edge[1], state["root"])         
        propose_state["tree"] = prop_edges_dict

    elif param_select == "tree":

        if move.__name__ == "rooted_NNI":
            prop_edges_dict, prop_post_order, hr, nodes_recompute, nodes_list = move(state[param_select].copy(), state["root"])
        else:
            prop_edges_dict, prop_post_order, hr = move(state[param_select].copy(), state["root"])

        propose_state["tree"] = prop_edges_dict
        propose_state["postorder"] = prop_post_order

    elif param_select == "srates":
        new_param, hr, pr_ratio = move(state[param_select])
        propose_state[param_select] = new_param
        current_rates = site_rates[:]        
        site_rates = get_siterates(new_param)


    if param_select == "tree":
        prop_mrca_list = get_mrca_list(propose_state["postorder"], cogset_taxa_list, state["root"])
#        if prop_mrca_list == mrca_list: print("mrca list did not change ", move.__name__)
#        assert len(prop_mrca_list) == len(mrca_list)
    else:
        prop_mrca_list = mrca_list[:]


    if move.__name__ == "scale_edge":
        for imr, mean_rate in enumerate(site_rates):
            old_edge_p_ts.append(state["transitionMat"][imr][change_edge].copy())
            state["transitionMat"][imr][change_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_edge]*mean_rate)

#        proposed_ll, proposed_llMat = cognateMatML2(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list)

#        proposed_ll = parCognateMatML(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, propose_state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list)

#        assert state["postorder"] == propose_state["postorder"]

        proposed_ll, proposed_llMat = cacheCognateMatML2(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, cache_LL_Mats, nodes_recompute, state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list, config.N_COG_CLASSES)

    elif move.__name__ == "node_slider":
        for imr, mean_rate in enumerate(site_rates):
            old_edge_p_ts.append(state["transitionMat"][imr][change_edge].copy())
            old_edge_p_t_as.append(state["transitionMat"][imr][change_parent_edge].copy())

            state["transitionMat"][imr][change_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_edge]*mean_rate)
            state["transitionMat"][imr][change_parent_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_parent_edge]*mean_rate)


#        proposed_ll = parCognateMatML(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list)


#        proposed_ll, proposed_llMat = cognateMatML2(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list)

        proposed_ll, proposed_llMat = cacheCognateMatML2(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, cache_LL_Mats, nodes_recompute, state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list, config.N_COG_CLASSES)

    elif move.__name__ == "rooted_NNI":
        for imr, mean_rate in enumerate(site_rates):
            state["transitionMat"][imr][nodes_list[0],nodes_list[3]], state["transitionMat"][imr][nodes_list[1],nodes_list[2]] = state["transitionMat"][imr][nodes_list[1],nodes_list[3]].copy(), state["transitionMat"][imr][nodes_list[0],nodes_list[2]].copy() #Perform a swap operation

#        proposed_ll = parCognateMatML(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, propose_state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list)


#        proposed_ll, proposed_llMat = cognateMatML2(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, propose_state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list)

        proposed_ll, proposed_llMat = cacheCognateMatML2(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, cache_LL_Mats, nodes_recompute, propose_state["postorder"], state["transitionMat"], config.N_CATS, prop_mrca_list, config.N_COG_CLASSES)

    else:
        prop_tmats = [get_prob_t(propose_state["pi"], propose_state["tree"], propose_state["rates"], mean_rate) for mean_rate in site_rates]

#        proposed_ll = parCognateMatML(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, propose_state["postorder"], prop_tmats, config.N_CATS, prop_mrca_list)

        proposed_ll, proposed_llMat = cognateMatML2(propose_state["pi"], state["root"], config.LEAF_LLMAT_LIST, propose_state["postorder"], prop_tmats, config.N_CATS, prop_mrca_list, config.N_COG_CLASSES)

    current_ll = state["logLikehood"]
    ll_ratio = proposed_ll - current_ll + pr_ratio
    ll_ratio = (1.0/T_i)*ll_ratio
    ll_ratio += hr

    if np.log(random.random()) <= ll_ratio:
        if param_select == "bl":
            state["tree"] = prop_edges_dict#propose_state["tree"]
        elif param_select == "tree":
            state[param_select] = prop_edges_dict#propose_state[param_select]
            state["postorder"] = prop_post_order
            cache_paths_dict = adjlist2reverse_nodes_dict(state[param_select])
        else:
            state[param_select] = new_param#propose_state[param_select]

        if move.__name__ not in ["scale_edge", "rooted_NNI", "node_slider"]:
            state["transitionMat"] = prop_tmats

        if move.__name__ == "rooted_NNI":
            for ip_t, p_t in enumerate(state["transitionMat"]):
                del state["transitionMat"][ip_t][nodes_list[0],nodes_list[2]], state["transitionMat"][ip_t][nodes_list[1],nodes_list[3]]

        state["logLikehood"] = proposed_ll
        cache_LL_Mats = proposed_llMat

        accepts_count[param_select,move.__name__] += 1

        mrca_list = prop_mrca_list[:]
    else:
        if param_select == "srates":
            site_rates = current_rates[:]
        elif move.__name__ == "scale_edge":
            for ip_t, p_t in enumerate(state["transitionMat"]):
                state["transitionMat"][ip_t][change_edge] = old_edge_p_ts[ip_t]
        elif move.__name__ == "node_slider":
            for ip_t, p_t in enumerate(state["transitionMat"]):
                state["transitionMat"][ip_t][change_edge] = old_edge_p_ts[ip_t]
                state["transitionMat"][ip_t][change_parent_edge] = old_edge_p_t_as[ip_t]
        elif move.__name__ == "rooted_NNI":
            for ip_t, p_t in enumerate(state["transitionMat"]):
                del state["transitionMat"][ip_t][nodes_list[0],nodes_list[3]], state["transitionMat"][ip_t][nodes_list[1],nodes_list[2]]
#        if param_select == "tree":
#            mrca_list = old_mrca_list[:]

    if n_iter % config.THIN == 0:
        TL = sum(state["tree"].values())
        stationary_freqs = "\t".join([str(state["pi"][idx]) for idx in range(config.N_CHARS)])
        sampled_tree = adjlist2newickBL(state["tree"], adjlist2nodes_dict(state["tree"]), state["root"])+";"
        print(n_iter, current_ll, proposed_ll, TL, param_select, move.__name__, T_i, sep="\t")
        print(n_iter, state["logLikehood"], TL, state["srates"], sep="\t", file=params_fileWriter)
        print(n_iter, sampled_tree, sep="\t", file=trees_fileWriter)
        if T_i < T_stop: break
        T_i = T_i * mult_scale       

    n_iter += 1

for k, v in moves_count.items():
    print(k, accepts_count[k], v)




