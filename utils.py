from collections import defaultdict

def readPhy(fname):
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
        char_vector = char_vector.replace(" ","")
        for ch in char_vector:
            if ch not in alphabet and ch not in ["?", "-"]:
                alphabet.append(ch)
        site_dict[taxa] = char_vector
        taxa_list.append(taxa)
    f.close()
    n_chars = len(alphabet)
    return n_leaves, n_chars, alphabet, site_dict, taxa_list, n_sites
    
def transform(site_dict):
    sites = defaultdict(lambda: defaultdict())
    for t, cvec in site_dict.items():
        for i, ch in enumerate(cvec):
            sites[i][t] = ch
    return sites
    
def print_mcmcstate(mcmc_state):
    for k, v in mcmc_state.items():
        print(k, v)
