from collections import defaultdict
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
    return n_leaves, n_chars, alphabet, site_dict, ll_mats,taxa_list, n_sites

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
    
cpdef sites2Mat(dict sites, int n_chars, list alphabet, list taxa_list):
    ll_mat = defaultdict(list)
    cdef int k_idx  
    for k, v in sites.items():
        for ch in v:
            if ch in ["?", "-"]:
                x = 0.000000001*np.ones(n_chars)
                #x = np.ones(n_chars)
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
        LL_MAT[k_idx] = np.array(v,order="F").T#np.ascontiguousarray(np.array(v).T, dtype=np.float32)
        #ll_mat[k] = np.array(v).T
        #print(k, np.array(v))
        #print(ll_mat[k].flags)
    return LL_MAT
    
    
    
