import sys, string
from collections import defaultdict

cog_cnt = 1
d = defaultdict(lambda: defaultdict(list))
d_t = defaultdict(list)
ALPHABET = list(string.ascii_letters)
langs = []
concepts = []

lines = open(sys.argv[1]).readlines()
for line in lines[2:]:
    if line.startswith("#"):
        continue
    arr = line.strip().split("\t")
    if arr[1] not in langs:
        langs.append(arr[1])
    if arr[2] not in concepts:
        concepts.append(arr[2])
    
    if len(d[arr[2]][arr[1]]) == 0 :
        d[arr[2]][arr[1]].append(int(arr[-1].replace("-","")))

for concept in d:
    states = []
    for s in d[concept].values():
        for t in s:
            states.append(t)
    #print(concept, states)
    min_state = min(states)
    for lang in langs:
        if lang not in d[concept]:
            d_t[lang].append("?")
        else:
            z = d[concept][lang]
            if len(z) == 1:
                d_t[lang].append(ALPHABET[z[0]-min_state])
            else:
                state_arr = []
                for x in d[concept][lang]:
                    state_arr.append(ALPHABET[x-min_state])
                d_t[lang].append("/".join(state_arr))
         
print(len(langs), len(concepts))
for lang, vector in d_t.items():
    #print(lang, vector)
    print(lang, " ".join(vector), sep="\t")
