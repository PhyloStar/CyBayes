#from __future__ import print_function
import argparse, utils, time
from mcmc_gamma import *
#from ML_gamma import *
from ML_scaled_cache import *
import config, sys
from collections import defaultdict
np.random.seed(1234)
random.seed(1234)

time_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Input a file in Phylip format with taxa and characters separated by a TAB character",  type=str)
parser.add_argument("-m", "--model", help="JC/F81/GTR",  type=str, default = "F81")
parser.add_argument("-n","--n_gen", help="Number of generations",  type=int, default = 100000)
parser.add_argument("-t","--thin", help="Number of generations after to print to screen",  type=int, default = 500)
parser.add_argument("-d","--data_type", help="Type of data if it is binary/multistate. Specify bin for binary and multi for multistate characters or phonetic alignments",  type=str)
parser.add_argument("-o","--output_file", help="Name of the out file prefix",  type=str, default = "temp")
parser.add_argument("-N","--n_chains", help="Name of the out file prefix",  type=int, default=4)
parser.add_argument("-dt","--deltaT", help="Chain temperature spacing",  type=float, default=0.1)
parser.add_argument("-a","--adjust_step", help="Chain temperature spacing",  type=int, default=500)
parser.add_argument("-C","--st_const", help="Upper bound",  type=int, default=10)
parser.add_argument("-p","--inc_temp", help="Harmonic/geometric spacing of temperature",  type=str, default="hm")
parser.add_argument("-cr","--converge_ratio", help="Convergence ratio",  type=float, default=1.5)
parser.add_argument("-ct","--cold_chain_thin", help="Thin the cold chain",  type=int, default=500)
parser.add_argument("-cns","--cold_n_samples", help="Number of samples in cold chain",  type=int, default = 1000)
parser.add_argument("-ncats","--nr_categories", help="Number of categories for discrete Gamma distribution",  type=int, default = 4)
parser.add_argument('--ml_scale', default=False, action='store_true')
parser.add_argument('--ml_cache', default=False, action='store_true')
#parser.add_argument("-sd","--seed", help="Initial Seed Value",  type=int, default = 1)
args = parser.parse_args()


if args.n_chains <= 1:
    print("Number of chains should be greater than 1")
    sys.exit(1)

if args.data_type == "bin":
    config.N_TAXA, config.N_CHARS, config.ALPHABET, site_dict, config.LEAF_LLMAT, config.TAXA, config.N_SITES = utils.readBinaryPhy(args.input_file)

elif args.data_type == "multi":
    config.N_TAXA, config.N_CHARS, config.ALPHABET, site_dict, config.LEAF_LLMAT, config.TAXA, config.N_SITES = utils.readMultiPhy(args.input_file)


config.IN_DTYPE = args.data_type
config.N_GEN = args.n_gen
config.THIN = args.thin
config.MODEL = args.model
config.N_NODES = 2*config.N_TAXA -1
config.N_CATS = args.nr_categories

print("Characters ", config.N_CHARS)
print("Sites ", config.N_SITES)
print("Number of TAXA ", config.N_TAXA)
print("Alphabet ", config.ALPHABET)
print("Number of Discrete Gamma Categories ", config.N_CATS)

n_rates = config.N_CHARS*(config.N_CHARS-1)//2
prior_pi = np.array([1]*config.N_CHARS)
prior_er = np.array([1]*n_rates)

if config.MODEL == "JC":
    config.NORM_BETA = config.N_CHARS/(config.N_CHARS-1)

init_state = state_init()
site_rates = get_siterates(init_state["srates"])

print("Frequencies ", np.array(init_state["pi"]))
print("Rates ", np.array(init_state["rates"]))
print("Site rates ", np.array(site_rates))
print("Discrete Gamma distn shape parameter", init_state["srates"])
print("Root of the tree", init_state["root"])

if args.ml_scale:
    print("Scaled Likelihood")
    init_state["logLikehood"], cache_LL_Mats, cache_rescale_list = matML_scaled(init_state["pi"], init_state["root"], config.LEAF_LLMAT, init_state["postorder"], init_state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)
else:
    print("Not scaled Likelihood")
    init_state["logLikehood"], _ = matML(init_state["pi"], init_state["root"], config.LEAF_LLMAT, init_state["postorder"], init_state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)

state = init_state.copy() #Why am I copying?
initial_tree = adjlist2newickBL(state["tree"], adjlist2nodes_dict(state["tree"]), state["root"])+";"
cache_paths_dict = adjlist2reverse_nodes_dict(state["tree"]) #for nodes recompute

#print("Initial Random Tree ", initial_tree, sep="\t") #Comment for small number of taxa

print("Initial Likelihood ",state["logLikehood"])

#Set the weights for MCMC moves
if config.MODEL == "F81":
    params_list = ["pi", "tree", "bl", "srates"]
    weights = np.array([1.0, 3, 4, 1.0], dtype=np.float64)
elif config.MODEL == "GTR":
    params_list = ["pi","rates", "tree", "bl", "srates"]
    weights = np.array([0.5, 1.0, 3, 4, 0.5], dtype=np.float64)
    if state["pi"].shape[0] == 2: #Hack the rates move in GTR model.
        weights[1] = 0.0
    print("Weights ", weights)
elif config.MODEL == "JC":
    params_list = ["tree", "bl", "srates"]
    weights = np.array([3, 4, 0.5], dtype=np.float64)

tree_move_weights = np.array([4, 1], dtype=np.float64)
bl_move_weights = np.array([3, 0, 2], dtype=np.float64)

weights = weights/np.sum(weights)
tree_move_weights = tree_move_weights/np.sum(tree_move_weights)
bl_move_weights = bl_move_weights/np.sum(bl_move_weights)

moves_count = defaultdict(int)
accepts_count = defaultdict(int)
moves_dict = {"pi": [mvDualSlider], "rates": [mvRatesSlider], "tree":[rooted_NNI, externalSPR], "bl":[scale_edge, node_slider, slide_edge], "srates":[mvShapeScaler]}

if args.data_type == "bin":
    moves_dict["pi"] = [mvBinaryDualSlider]

params_fileWriter = open(args.output_file+".log","w")
trees_fileWriter = open(args.output_file+".trees","w")
const_states = ["pi("+idx+")" for idx in config.ALPHABET]

propose_state, prop_tmats = {}, []

# Log-psuedo prior for different chains. Will be tuned later.
log_psuedo_prior = np.zeros(args.n_chains)
counts_states = np.zeros(args.n_chains)
global_counts_states = np.zeros(args.n_chains)
print("Log_psuedo_prior ", log_psuedo_prior)

#sys.exit(1)

if args.inc_temp == "hm":
    chain_T_dict = {i: 1.0/(1+(i*args.deltaT)) for i in range(args.n_chains)}#A harmonic spacing temperature chain.
elif args.inc_temp == "gm":
    chain_T_dict = {i: (1+args.deltaT)**-i for i in range(args.n_chains)}#Geometric spacing
print(chain_T_dict)
current_chain = 0 # Initialize with random state. np.random.randint(args.n_chains)
curr_beta_t = chain_T_dict[current_chain]

print("Iter", "LnL", "TL", "Alpha", *const_states, sep="\t", file=params_fileWriter)

n_iter, n_iter_chain0 = 1, 1

while(1):

    counts_states[current_chain] += 1

    if np.random.random() <=0.5 or n_iter == 1:
        propose_state["pi"],  propose_state["rates"], propose_state["srates"], propose_state["tree"], propose_state["postorder"] = state["pi"].copy(), state["rates"].copy(), state["srates"], state["tree"].copy(), state["postorder"].copy()
        
        current_ll, proposed_ll, ll_ratio, hastings_ratio, change_edge, pr_ratio, old_edge_p_ts, old_edge_p_t_as = 0.0, 0.0, 0.0, 0.0, None, 0.0, [], []
            
        param_select = np.random.choice(params_list, p=weights)

        if param_select == "tree":
            move = np.random.choice(moves_dict[param_select], p=tree_move_weights)
        elif param_select == "bl":
            move = np.random.choice(moves_dict[param_select], p=bl_move_weights)
        else:
            move = np.random.choice(moves_dict[param_select])
        
        moves_count[param_select,move.__name__] += 1
        
        if param_select in ["pi", "rates"]:
            propose_param, hastings_ratio = move(state[param_select].copy())
            propose_state[param_select] = propose_param

        elif param_select == "bl":

            if move.__name__ == "scale_edge" or move.__name__ == "slide_edge":
                prop_edges_dict, hastings_ratio, pr_ratio, change_edge = move(state["tree"].copy())

            elif move.__name__ == "node_slider":
                prop_edges_dict, hastings_ratio, pr_ratio, change_edge, change_parent_edge = move(state["tree"].copy(), state["root"])

            propose_state["tree"] = prop_edges_dict

            nodes_recompute = get_path2root(cache_paths_dict, change_edge[1], state["root"]) #Recompute only for these nodes

        elif param_select == "tree":

            if move.__name__ == "rooted_NNI":
                prop_edges_dict, prop_post_order, hastings_ratio, nodes_recompute, nodes_list = move(state[param_select].copy(), state["root"])
            else:
                prop_edges_dict, prop_post_order, hastings_ratio = move(state[param_select].copy(), state["root"])

            propose_state["tree"] = prop_edges_dict
            propose_state["postorder"] = prop_post_order

        elif param_select == "srates":
            propose_param, hastings_ratio, pr_ratio = move(state[param_select])
            propose_state[param_select] = propose_param
            current_rates = site_rates[:]        
            site_rates = get_siterates(propose_param)

        #cache_rescale_list is refreshed when performing a tree move or substitution model parameters moves

        if move.__name__ == "scale_edge" or move.__name__ == "slide_edge":
            for imr, mean_rate in enumerate(site_rates):
                old_edge_p_ts.append(state["transitionMat"][imr][change_edge].copy())
                state["transitionMat"][imr][change_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_edge]*mean_rate)

            if args.ml_scale:
                if not args.ml_cache:
                    proposed_ll, proposed_llMat, prop_rescale_list = matML_scaled(propose_state["pi"],  state["root"], config.LEAF_LLMAT, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)
                else:
                    proposed_ll, proposed_llMat, prop_rescale_list = matML_scaled_cache(propose_state["pi"],  state["root"], config.LEAF_LLMAT, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS, cache_LL_Mats, cache_rescale_list, nodes_recompute)
                
            else:
                proposed_ll, _ = matML(propose_state["pi"],  state["root"], config.LEAF_LLMAT, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)

        elif move.__name__ == "node_slider":
            for imr, mean_rate in enumerate(site_rates):
                old_edge_p_ts.append(state["transitionMat"][imr][change_edge].copy())
                old_edge_p_t_as.append(state["transitionMat"][imr][change_parent_edge].copy())

                state["transitionMat"][imr][change_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_edge]*mean_rate)
                state["transitionMat"][imr][change_parent_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_parent_edge]*mean_rate)

            if args.ml_scale:
#                proposed_ll, proposed_llMat, prop_rescale_list = matML_scaled(propose_state["pi"],  state["root"], config.LEAF_LLMAT, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)
                if not args.ml_cache:
                    proposed_ll, proposed_llMat, prop_rescale_list = matML_scaled(propose_state["pi"],  state["root"], config.LEAF_LLMAT, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)
                else:
                    proposed_ll, proposed_llMat, prop_rescale_list = matML_scaled_cache(propose_state["pi"],  state["root"], config.LEAF_LLMAT, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS, cache_LL_Mats, cache_rescale_list, nodes_recompute)
            else:
                proposed_ll, _ = matML(propose_state["pi"],  state["root"], config.LEAF_LLMAT, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)

        else:
            prop_tmats = [get_prob_t(propose_state["pi"], propose_state["tree"], propose_state["rates"], mean_rate) for mean_rate in site_rates]

            if args.ml_scale:
                proposed_ll, proposed_llMat, prop_rescale_list = matML_scaled(propose_state["pi"],  state["root"], config.LEAF_LLMAT, propose_state["postorder"], prop_tmats, config.N_SITES, config.N_TAXA, config.N_CATS)
            else:
                proposed_ll, _ = matML(propose_state["pi"],  state["root"], config.LEAF_LLMAT, propose_state["postorder"], prop_tmats, config.N_SITES, config.N_TAXA, config.N_CATS)


        current_ll = state["logLikehood"]
        ll_ratio = proposed_ll - current_ll + pr_ratio
        ll_ratio = curr_beta_t*ll_ratio
        ll_ratio += hastings_ratio

        if np.log(np.random.random()) <= ll_ratio:
            if param_select == "bl":
                state["tree"] = prop_edges_dict#propose_state["tree"]
            elif param_select == "tree":
                state[param_select] = prop_edges_dict#propose_state[param_select]
                state["postorder"] = prop_post_order
                cache_paths_dict = adjlist2reverse_nodes_dict(state[param_select])
            else:
                state[param_select] = propose_param#propose_state[param_select]
#                if param_select == "srates":
#                    print(param_select, propose_param)

            if move.__name__ not in ["scale_edge", "node_slider", "slide_edge"]:
                state["transitionMat"] = prop_tmats

            state["logLikehood"] = proposed_ll

            accepts_count[param_select,move.__name__] += 1

            cache_LL_Mats = proposed_llMat
            cache_rescale_list = prop_rescale_list

        else:
            if param_select == "srates":
                site_rates = current_rates[:]
            elif move.__name__ == "scale_edge" or move.__name__ == "slide_edge":
                for ip_t, p_t in enumerate(state["transitionMat"]):
                    state["transitionMat"][ip_t][change_edge] = old_edge_p_ts[ip_t]
            elif move.__name__ == "node_slider":
                for ip_t, p_t in enumerate(state["transitionMat"]):
                    state["transitionMat"][ip_t][change_edge] = old_edge_p_ts[ip_t]
                    state["transitionMat"][ip_t][change_parent_edge] = old_edge_p_t_as[ip_t]

    else:
        jump_ratio = 0

        if current_chain == 0:
            propose_chain = 1
        elif current_chain == args.n_chains-1:
            propose_chain = current_chain - 1
        else:
            propose_chain = random.choice([current_chain-1, current_chain+1])
#            if propose_chain > current_chain:
#                jump_ratio = np.log(0.5)
#            else:
#                jump_ratio = np.log(2.0)

        diff_beta = chain_T_dict[propose_chain]-chain_T_dict[current_chain]

        swap_ratio = (diff_beta*(state["logLikehood"])) + log_psuedo_prior[propose_chain] - log_psuedo_prior[current_chain] + jump_ratio
        
        moves_count["jump:", current_chain, propose_chain] += 1

        if np.log(np.random.random()) <= swap_ratio:
            accepts_count["jump:", current_chain, propose_chain] += 1
            if n_iter % config.THIN == 0:
                print("Accepted Current chain ", current_chain, "Proposed chain ", propose_chain, "with swap ratio ", swap_ratio, "at iteration", n_iter)
            current_chain = propose_chain
            curr_beta_t = chain_T_dict[current_chain]

    TL = sum(state["tree"].values())
    stationary_freqs = "\t".join([str(state["pi"][idx]) for idx in range(config.N_CHARS)])
    sampled_tree = adjlist2newickBL(state["tree"], adjlist2nodes_dict(state["tree"]), state["root"])+";"
    if n_iter % config.THIN == 0:
        print(n_iter, current_ll, proposed_ll, TL, state["srates"], param_select, move.__name__, current_chain, sep="\t")

    if current_chain == 0:
        n_iter_chain0 += 1
        if n_iter_chain0 % args.cold_chain_thin == 0:
            print(n_iter_chain0, state["logLikehood"], TL, state["srates"], stationary_freqs, sep="\t", file=params_fileWriter)
            print(n_iter_chain0, sampled_tree, sep="\t", file=trees_fileWriter)
#            print(n_iter_chain0, state["logLikehood"], TL, state["srates"], stationary_freqs, sep="\t")

    n_iter +=  1
    global_counts_states[current_chain] += 1
        
    #Adjust logpsuedo prior according to Geyer et al 2011
    if n_iter % args.adjust_step == 0:

        temp_counts = counts_states/np.sum(counts_states)

        max_temp_counts = np.max(global_counts_states)
        min_temp_counts = np.min(global_counts_states)

#        print("Visited chains relative frequency ", counts_states, temp_counts, min_temp_counts, max_temp_counts)
        counts_states = np.zeros(args.n_chains) #May be zero the counts if needed. Let us check later 

        if min_temp_counts > 0 and max_temp_counts/min_temp_counts < args.converge_ratio and n_iter_chain0 > args.cold_chain_thin * args.cold_n_samples:
#        if min_temp_counts > 0 and max_temp_counts/min_temp_counts < args.converge_ratio:
            break
        
        clip_x = np.clip(np.log(np.max(temp_counts)/temp_counts), None, args.st_const) #c_k = 1/d_k. 
#        clip_x = np.clip(np.log(1/temp_counts), None, args.st_const)
#        print("Clipped Log-pseudo prior ",clip_x)

        log_psuedo_prior += clip_x
#        print("Log Psuedo Prior before subtracting min", log_psuedo_prior)
        log_psuedo_prior = log_psuedo_prior-np.min(log_psuedo_prior)
#        print("Log Psuedo Prior after subtracting min", log_psuedo_prior)

for k, v in moves_count.items():
    print(*k, round(accepts_count[k]/v, 2))

time_end = time.time()

print("MCMC chain run for {} iterations for {} seconds".format(n_iter, round(time_end-time_start, 3)))

print("Chain temperature visit counts ",*global_counts_states, sep="\t")

print("Log Psuedo Prior ", log_psuedo_prior)

