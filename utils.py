from collections import defaultdict
import numpy as np

def readBinaryPhy(fname):
    site_dict = defaultdict()
    alphabet, taxa_list = [], []
    f = open(fname)
    header = f.readline().strip()
    n_leaves, n_sites = map(int,header.split(" "))
    
    for line in f:
        if len(line.strip()) < 1:
            continue
        taxa, char_vector = line.strip().split()
        taxa = taxa.replace(" ","")
        assert len(char_vector) == n_sites
        for ch in char_vector:
            if ch not in alphabet and ch not in ["?", "-"]:
                alphabet.append(ch)
        site_dict[taxa] = char_vector
        taxa_list.append(taxa)
    f.close()
    n_chars = len(alphabet)
    ll_mats= sites2Mat(site_dict, n_chars, alphabet)
    return n_leaves, n_chars, alphabet, site_dict, ll_mats,taxa_list, n_sites

def readPhy(fname):
    site_dict = defaultdict()
    alphabet, taxa_list = [], []
    f = open(fname)
    header = f.readline().strip()
    n_leaves, n_sites = map(int,header.split(" "))
    
    for line in f:
        if len(line.strip()) < 1:
            continue
        taxa, char_vector = line.strip().split("\t")
        taxa = taxa.replace(" ","")
        #char_vector = char_vector.replace(" ","")

        for ch in char_vector.split(" "):
            temp_ch = ch.split("/")
            for tch in temp_ch:
                if tch not in alphabet and tch not in ["?", "-"]:
                    alphabet.append(tch)
        site_dict[taxa] = char_vector.split(" ")
        taxa_list.append(taxa)
    f.close()
    n_chars = len(alphabet)
    ll_mats= sites2Mat(site_dict, n_chars, alphabet)
    return n_leaves, n_chars, alphabet, site_dict, ll_mats,taxa_list, n_sites
    
def transform(site_dict):
    sites = defaultdict(lambda: defaultdict())
    for t, cvec in site_dict.items():
        for i, ch in enumerate(cvec):
            sites[i][t] = ch
    return sites
    
def print_mcmcstate(mcmc_state):
    for k, v in mcmc_state.items():
        print(k, v)

def sites2Mat(sites, n_chars, alphabet):
    ll_mat = defaultdict(list)
    for k, v in sites.items():
        for ch in v:
            if ch in ["?", "-"]:
                x = np.ones(n_chars)
            elif "/" in ch:
                y = ch.split("/")
                x = np.zeros(n_chars)
                for t in y:
                    idx = alphabet.index(t)
                    x[idx] = 1.0
            else:
                x = np.zeros(n_chars)
                idx = alphabet.index(ch)
                x[idx] = 1.0
            ll_mat[k].append(x)

    for k, v in ll_mat.items():
        ll_mat[k] = np.array(v).T#np.ascontiguousarray(np.array(v).T, dtype=np.float32)
        #print(k, np.array(v))
        #print(ll_mat[k].flags)
    return ll_mat
    
    
    
