#from __future__ import print_function
import argparse, utils, time
from mcmc_gamma import *
from ML_gamma import *
import config
from collections import defaultdict
np.random.seed(1234)
random.seed(1234)
#global N_TAXA, N_CHARS, ALPHABET, LEAF_LLMAT, TAXA, MODEL, IN_DTYPE, NORM_BETA

time_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="Input a file in Phylip format with taxa and characters separated by a TAB character",  type=str)
parser.add_argument("-m", "--model", help="JC/F81/GTR",  type=str, default = "F81")
parser.add_argument("-n","--n_gen", help="Number of generations",  type=int, default = 100000)
parser.add_argument("-t","--thin", help="Number of generations after to print to file",  type=int, default = 500)
parser.add_argument("-d","--data_type", help="Type of data if it is binary/multistate. Multistate characters should be separated by a space whereas binary need not be. Specify bin for binary and multi for multistate characters or phonetic alignments",  type=str)
parser.add_argument("-o","--output_file", help="Name of the out file prefix",  type=str)
parser.add_argument("-N","--n_chains", help="Name of the out file prefix",  type=int, default=4)
parser.add_argument("-dt","--deltaT", help="Chain temperature spacing",  type=float, default=0.1)
parser.add_argument("-a","--adjust_step", help="Chain temperature spacing",  type=int, default=100)
parser.add_argument("-C","--st_const", help="Upper bound",  type=int, default=10)

args = parser.parse_args()

if args.data_type == "bin":
    config.N_TAXA, config.N_CHARS, config.ALPHABET, site_dict, config.LEAF_LLMAT, config.TAXA, config.N_SITES = utils.readBinaryPhy(args.input_file)

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
site_rates = get_siterates(init_state["srates"])

cache_LL_Mats, cache_paths_dict = None, None
init_state["logLikehood"], cache_LL_Mats = matML(init_state["pi"], init_state["root"], config.LEAF_LLMAT, init_state["postorder"], init_state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)
state = init_state.copy()
init_tree = adjlist2newickBL(state["tree"], adjlist2nodes_dict(state["tree"]), state["root"])+";"
cache_paths_dict = adjlist2reverse_nodes_dict(state["tree"])

print("Initial Random Tree ", init_tree, sep="\t")

print("Initial Likelihood ",state["logLikehood"])

if config.MODEL == "F81":
    params_list = ["pi", "tree", "bl", "srates"]
    weights = np.array([0.5, 3, 4, 0.5], dtype=np.float64)
elif config.MODEL == "GTR":
    params_list = ["pi","rates", "tree", "bl", "srates"]
    weights = np.array([0.5, 0.5, 3, 4, 0.5], dtype=np.float64)
elif config.MODEL == "JC":
    params_list = ["bl", "tree", "srates"]
    weights = np.array([4, 3, 0.5], dtype=np.float64)

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

propose_state, prop_tmats = {}, []

# Log-psuedo prior for different chains. Will be tuned later.
log_psuedo_prior = np.zeros(args.n_chains)
counts_states = np.zeros(args.n_chains)
global_counts_states = np.zeros(args.n_chains)
print("Log_psuedo_prior ", log_psuedo_prior)

chain_T_dict = {i: 1.0/(1+(i*args.deltaT)) for i in range(args.n_chains)}#A harmonic spacing temperature chain. Add a parser setting here.
chain_T_dict = {i: 1.0/(1+args.deltaT)**i for i in range(args.n_chains)}#Geometric spacing
print(chain_T_dict)
current_chain = 0 # Initialize with random state. np.random.randint(args.n_chains)
curr_beta_t = chain_T_dict[current_chain]

print("Iter", "LnL", "TL", "Alpha", sep="\t", file=params_fileWriter)
#for n_iter in range(1,  config.N_GEN+1):
n_iter = 1
while(1):

    counts_states[current_chain] += 1

    if random.random() <=0.5:
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
        
        if move.__name__ == "scale_edge":
            for imr, mean_rate in enumerate(site_rates):
                old_edge_p_ts.append(state["transitionMat"][imr][change_edge].copy())
                state["transitionMat"][imr][change_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_edge]*mean_rate)

            proposed_ll, proposed_llMat = cache_matML(propose_state["pi"], state["root"], config.LEAF_LLMAT, cache_LL_Mats, nodes_recompute, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)

        elif move.__name__ == "node_slider":
            for imr, mean_rate in enumerate(site_rates):
                old_edge_p_ts.append(state["transitionMat"][imr][change_edge].copy())
                old_edge_p_t_as.append(state["transitionMat"][imr][change_parent_edge].copy())

                state["transitionMat"][imr][change_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_edge]*mean_rate)
                state["transitionMat"][imr][change_parent_edge] = get_edge_transition_mat(propose_state["pi"], propose_state["rates"], propose_state["tree"][change_parent_edge]*mean_rate)

            proposed_ll, proposed_llMat = cache_matML(propose_state["pi"], state["root"], config.LEAF_LLMAT, cache_LL_Mats, nodes_recompute, state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)

        elif move.__name__ == "rooted_NNI":
            for imr, mean_rate in enumerate(site_rates):
                state["transitionMat"][imr][nodes_list[0],nodes_list[3]], state["transitionMat"][imr][nodes_list[1],nodes_list[2]] = state["transitionMat"][imr][nodes_list[1],nodes_list[3]].copy(), state["transitionMat"][imr][nodes_list[0],nodes_list[2]].copy()

            proposed_ll, proposed_llMat = cache_matML(propose_state["pi"], state["root"], config.LEAF_LLMAT, cache_LL_Mats, nodes_recompute, propose_state["postorder"], state["transitionMat"], config.N_SITES, config.N_TAXA, config.N_CATS)

        else:
            prop_tmats = [get_prob_t(propose_state["pi"], propose_state["tree"], propose_state["rates"], mean_rate) for mean_rate in site_rates]
            proposed_ll, proposed_llMat = matML(propose_state["pi"],  state["root"], config.LEAF_LLMAT, propose_state["postorder"], prop_tmats, config.N_SITES, config.N_TAXA, config.N_CATS)

        #prop_tmats = [get_prob_t(propose_state["pi"], propose_state["tree"], propose_state["rates"], mean_rate) for mean_rate in site_rates]
        #proposed_ll, proposed_llMat = matML(propose_state["pi"],  state["root"], config.LEAF_LLMAT, propose_state["postorder"], prop_tmats)

        current_ll = state["logLikehood"]
        ll_ratio = proposed_ll - current_ll + pr_ratio
        ll_ratio = curr_beta_t*ll_ratio
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

            #state["transitionMat"] = prop_tmats#Comment this for caching

            accepts_count[param_select,move.__name__] += 1
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

    else:
        jump_ratio = 0
        list_n_chains = list(range(args.n_chains))
        list_n_chains.remove(current_chain)
        propose_chain = random.choice(list_n_chains)

#        if current_chain == 0:
#            propose_chain = 1
#        elif current_chain == args.n_chains-1:
#            propose_chain = current_chain - 1
#        else:
#            propose_chain = random.choice([current_chain-1, current_chain+1])
#            if propose_chain > current_chain:
#                jump_ratio = np.log(0.5)
#            else:
#                jump_ratio = np.log(2.0)

        diff_beta = chain_T_dict[propose_chain]-chain_T_dict[current_chain]

        swap_ratio = (diff_beta*(state["logLikehood"])) + log_psuedo_prior[propose_chain] - log_psuedo_prior[current_chain] + jump_ratio
        
        moves_count["jump"] += 1

        if np.log(random.random()) <= swap_ratio:
            if n_iter % config.THIN == 0:
                print("Accepted Current chain ", current_chain, "Proposed chain ", propose_chain, "with swap ratio ", swap_ratio, "at iteration", n_iter)
            current_chain = propose_chain
            curr_beta_t = chain_T_dict[current_chain]
            accepts_count["jump"] += 1

    TL = sum(state["tree"].values())
#    stationary_freqs = "\t".join([str(state["pi"][idx]) for idx in range(config.N_CHARS)])
    sampled_tree = adjlist2newickBL(state["tree"], adjlist2nodes_dict(state["tree"]), state["root"])+";"
    if n_iter % config.THIN == 0:
        print(n_iter, current_ll, proposed_ll, TL, param_select, move.__name__, current_chain, sep="\t")
    if current_chain == 0:
        print(n_iter, state["logLikehood"], TL, state["srates"], sep="\t", file=params_fileWriter)
        print(n_iter, sampled_tree, sep="\t", file=trees_fileWriter)

    #Adjust logpsuedo prior
    if n_iter % args.adjust_step == 0:
        temp_counts = counts_states/np.sum(counts_states)
        
        max_temp_counts = np.max(temp_counts)
        min_temp_counts = np.min(temp_counts)
#        print("Visited chains relative frequency ", counts_states, temp_counts, min_temp_counts, max_temp_counts)
        counts_states = np.zeros(args.n_chains) #May be zero the counts if needed. Let us check later 
        if (min_temp_counts > 0 and max_temp_counts/min_temp_counts < 1.2) and n_iter > args.n_gen: break
        
        clip_x = np.clip(np.log(np.max(temp_counts)/temp_counts), None, args.st_const) #c_k = 1/d_k. 
#        clip_x = np.clip(np.log(1/temp_counts), None, args.st_const)
#        print("Clipped Log-pseudo prior ",clip_x)

        log_psuedo_prior += clip_x
#        print("Log Psuedo Prior before subtracting min", log_psuedo_prior)
        log_psuedo_prior = log_psuedo_prior-np.min(log_psuedo_prior)
#        print("Log Psuedo Prior after subtracting min", log_psuedo_prior)
        #Make the cliiping bound as input variable
        
    n_iter +=  1
    global_counts_states[current_chain] += 1

for k, v in moves_count.items():
    print(k, accepts_count[k], v)

time_end = time.time()

print("MCMC chain run for {} iterations for {} seconds".format(n_iter, round(time_end-time_start, 3)))

print(global_counts_states)

