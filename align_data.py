from lingpy import *
from lingpy.sequence.sound_classes import token2class
from sys import argv
from collections import defaultdict

map_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"]

if argv[1].startswith('Mayan') or argv[1].startswith('Mixe'):
    rc(schema='asjp')
else:
    rc(schema='ipa')

lang2glottcode = defaultdict()

lines = open("langs_glottocode_list.txt","r").readlines()
for line in lines[1:]:
    l, g = line.replace("\n","").split("\t")
    if len(g) < 1: continue
    lang2glottcode[l] = g

# first do only concepts
wl = Wordlist(argv[1])
wl.add_entries('tokens','FORM', ipa2tokens, merge_vowels=True,
        semi_diacritics='sh')

# check for bad cognate set alignment
outfname = argv[1].split("/")[-1].replace('.tsv', '')
if 'cognates' in argv:
    outfname += '_cognates'
    lex = LexStat(wl)
    #lex.cluster(method='sca', threshold=0.45)
    alm = Alignments(lex, ref='cogid', transcription='FORM')
    target = 'cogid'
else:
    target = 'concept'
    alm = Alignments(wl, ref='concept', transcription='FORM')
if "library" in argv:
    alm.align(method='library', iterate=False)
    outfname += '_library'
else:
    alm.align(method='progressive', iterate=False)
    outfname += '_prog'

# assemble data for each alignment
uniq_chars = []
phylip = {}
for lang in alm.cols:
    phylip[lang] = ''
for msa, vals in alm.msa[target].items():

    langs = vals['taxa']
    seqs = vals['alignment']

    print(langs)
    alm_len = len(seqs[0])
    
    for i, lang in enumerate(alm.cols):
        raxml_alm_str = ""
        if lang not in langs:
            alm_str = alm_len * '?'
            raxml_alm_str = alm_str
        else:
            alm_str = ''.join([token2class(x, 'sca') if x != '-' else '-' for x in seqs[langs.index(lang)]])
            for ch in alm_str:
                if ch == '-':
                    continue
                if ch not in uniq_chars:
                    uniq_chars.append(ch)
            raxml_alm_str = ''.join([map_chars[uniq_chars.index(x)] if x != '-' else '-' for x in alm_str])
            #print raxml_alm_str
        
#        phylip[lang] += raxml_alm_str
        phylip[lang] += alm_str

print(len(uniq_chars)," characters in alphabet")
print(sorted(uniq_chars))

glottcode_list, n_chars = [], None

for tax in alm.cols:
#    if tax not in lang2glottcode: continue
#    gcode = lang2glottcode[tax]
    if tax not in glottcode_list:
        glottcode_list.append(tax)
#        phylip[gcode] = phylip[tax]
    n_chars = len(phylip[tax])

with open(outfname+'.phy', 'w') as f:
    f.write(str(len(glottcode_list))+" "+str(n_chars)+"\n")
    for gcode in glottcode_list:
        f.write('{0:40}{1}'.format(gcode, phylip[gcode])+'\n')
        
        

