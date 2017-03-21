import sys, string
from collections import defaultdict

cog_cnt = 1
d = defaultdict(lambda: defaultdict())
d_t = defaultdict(list)
ALPHABET = list(string.ascii_uppercase)
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
    d[arr[2]][arr[1]] = int(arr[-1].replace("-",""))

for concept in d:
    states = list(d[concept].values())
    #print(concept, len(states))
    min_state = min(states)
    for lang in langs:
        if lang not in d[concept]:
            d_t[lang].append("?")
        else:
            d_t[lang].append(ALPHABET[d[concept][lang]-min_state])
         
print(len(langs), len(concepts))
for lang, vector in d_t.items():
    print(lang.ljust(40), "".join(vector), sep="\t")
