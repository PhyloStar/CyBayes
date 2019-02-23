from collections import defaultdict
import os
os.environ["OMP_NUM_THREADS"] = '1' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1' # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = '1' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = '1' # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = '1' # export NUMEXPR_NUM_THREADS=6

import numpy as np

cpdef readBinaryPhy(str fname):
    site_dict = {}#defaultdict()
    cdef list alphabet, taxa_list
    cdef int n_leaves, n_sites, n_chars
    cdef str line, taxa, char_vector, ch, header
    
    alphabet, taxa_list = [], []
    f = open(fname)
    header = f.readline().strip()
    n_leaves, n_sites = map(int,header.split(" "))
    alphabet = ["0", "1"]
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
    ll_mats= sites2Mat(site_dict, n_chars, alphabet, taxa_list)
    #print(ll_mats)
    return n_leaves, n_chars, alphabet, site_dict, ll_mats,taxa_list, n_sites

cpdef readCognatePhy(str fname):
    ll_mats_list, alphabet, taxa_list, n_leaves = [], [], [], 0

    f = open(fname)

    site_dict, sites_dict_list, cogset_taxa_list = {}, [], []
    
    for line in f:
        if line == "\n":
            sites_dict_list.append(site_dict)
            site_dict = {}
            continue
#        print(line)
        taxa, char_vector = line.strip().split("\t")
        taxa = taxa.replace(" ","")

        for ch in char_vector:
            if ch not in alphabet and ch not in ["?", "-"]:
                alphabet.append(ch)
        site_dict[taxa] = list(char_vector)

        if taxa not in taxa_list:
            taxa_list.append(taxa)

    n_chars = len(alphabet)

    taxa_list = sorted(taxa_list)

    for i in range(0, len(sites_dict_list)):
        site_dict = sites_dict_list[i]

        tax_list = list(site_dict.keys())

        if len(tax_list) > 1:
            ncols = len(list(site_dict.values())[0])

            cogset_taxa_list.append([taxa_list.index(k)+1 for k in tax_list])
            sd = {}
            for lang in taxa_list:
                if lang not in site_dict:
                    gap_row = ["?"]*ncols
                    sd[lang] = gap_row
                else:
                    sd[lang] = site_dict[lang]
                
            ll_mats = sites2Mat(sd, n_chars, alphabet, taxa_list)
            ll_mats_list.append(ll_mats) 

    f.close()    

    n_leaves = len(taxa_list)
    return n_leaves, n_chars, alphabet, ll_mats_list, taxa_list, cogset_taxa_list
    

def readPhy(fname):
    site_dict = {}#defaultdict()
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
        print(taxa, char_vector)
        for ch in char_vector.split(" "):
            temp_ch = ch.split("/")
            for tch in temp_ch:
                if tch not in alphabet and tch not in ["?", "-"]:
                    alphabet.append(tch)
        site_dict[taxa] = char_vector.split(" ")
        taxa_list.append(taxa)
    f.close()
    n_chars = len(alphabet)
    
    ll_mats= sites2Mat(site_dict, n_chars, alphabet, taxa_list)
    
    return n_leaves, n_chars, alphabet, site_dict, ll_mats, taxa_list, n_sites

def readMultiPhy(fname):
    site_dict = {}#defaultdict()
    alphabet, taxa_list = [], []
    f = open(fname)
    header = f.readline().strip()
    n_leaves, n_sites = map(int,header.split(" "))
    
    for line in f:
        if len(line.strip()) < 1:
            continue
        #taxa, char_vector = line.strip().split("\t")
        taxa, char_vector = line.strip().split()
        taxa = taxa.replace(" ","")
        #char_vector = char_vector.replace(" ","")
        #print(taxa, char_vector)
        for ch in char_vector:#.split(" "):
            if ch not in alphabet and ch not in ["?", "-"]:
                alphabet.append(ch)
        site_dict[taxa] = list(char_vector)
        taxa_list.append(taxa)
    f.close()
    n_chars = len(alphabet)
    
    ll_mats= sites2Mat(site_dict, n_chars, alphabet, taxa_list)
    
    return n_leaves, n_chars, alphabet, site_dict, ll_mats, taxa_list, n_sites

cpdef sites2Mat(dict sites, int n_chars, list alphabet, list taxa_list):
    ll_mat = defaultdict(list)
    cdef int k_idx  
    for k, v in sites.items():
        for ch in v:
            if ch in ["?", "-"]:
                x = np.ones(n_chars)
                #x = np.ones(n_chars)/n_chars
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
    cdef dict LL_MAT = {}
    for k, v in ll_mat.items():
        k_idx = taxa_list.index(k)+1
        LL_MAT[k_idx] = np.array(v, order="F").T#np.ascontiguousarray(np.array(v).T, dtype=np.float32)
        #ll_mat[k] = np.array(v).T
        #print(k, np.array(v))
        #print(ll_mat[k].flags)
    return LL_MAT
    
    
    
