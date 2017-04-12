from lingpy import *
from lingpy.sequence.sound_classes import token2class
from sys import argv
from collections import defaultdict

map_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"]

if argv[1].startswith('Mayan') or argv[1].startswith('Mixe'):
    rc(schema='asjp')
else:
    rc(schema='ipa')

# first do only concepts
wl = Wordlist(argv[1])
wl.add_entries('tokens', 'transcription', ipa2tokens, merge_vowels=True,
        semi_diacritics='sh')

# check for bad cognate set alignment
if 'cognates' in argv:
    outfname = argv[1].replace('.tsv', '') + '_cognates'
    lex = LexStat(wl)
    lex.cluster(method='sca', threshold=0.45)
    alm = Alignments(lex, ref='scaid', transcription='transcription')
    target = 'scaid'
else:
    outfname = argv[1].replace('.tsv', '')
    target = 'concept'
    alm = Alignments(wl, ref='concept', transcription='transcription')
alm.align(method='progressive', iterate=False)

# assemble data for each alignment
uniq_chars = []
phylip = defaultdict(list)
len_alms = 0
for lang in alm.cols:
    phylip[lang] = []
for msa, vals in alm.msa[target].items():

    langs = vals['taxa']
    seqs = vals['alignment']

    alm_len = len(seqs[0])
    len_alms += alm_len
    #print alm_len
    for i, lang in enumerate(alm.cols):
        raxml_alm_str = ""
        if lang not in langs:
            alm_str = alm_len * '?'
            raxml_alm_str = list(alm_str)
        else:
            raxml_alm_str = [token2class(x, 'sca') if x != '-' else '-' for x in seqs[langs.index(lang)]]
            for ch in raxml_alm_str:
                if ch == '-':
                    continue
                if ch not in uniq_chars:
                    uniq_chars.append(ch)
            #raxml_alm_str = ' '.join([map_chars[uniq_chars.index(x)] if x != '-' else '-' for x in alm_str])
            #print raxml_alm_str

        phylip[lang] += raxml_alm_str
#        phylip[lang] += alm_str

print len(uniq_chars)," ALPHABET"
print sorted(uniq_chars)
with open(outfname+'.prog.phy', 'w') as f:
    f.write(str(len(phylip.keys()))+" "+str(len_alms)+"\n")
    for tax in alm.cols:
        f.write(tax+"\t"+" ".join(phylip[tax])+"\n")
        #f.write('{0:40}{1}'.format(tax, phylip[tax])+'\n')
        

