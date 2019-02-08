'''python3 nexus2phylip.py data-aa-58-200.paps.nex
'''
import sys

inname = sys.argv[1]

outname = open(inname.split("/")[-1].replace(".nex", ".phy"), "w")

taxa_list, chars_list = [], []
ntaxa, nchars = 0, 0

for line in open(inname, "r"):
    line = line.strip("\n")
    if line[-1] == ";" or "matrix" in line: continue
    arr = line.split()
    if len(arr) > 2:
        print("Whitespace character in taxa name ", file=sys.stderr)
        taxa = "_".join(arr[:-1])
        chars = arr[-1]
    else:
        taxa, chars = arr

    taxa_list.append(taxa)
    chars_list.append(chars)
    if nchars > 0 and nchars != len(chars): 
        print("Unequal number of characters in taxa {}".format(taxa), file=sys.stderr)
        sys.exit(1)
    else:
        nchars = len(chars)

ntaxa = len(taxa_list)
print(ntaxa, nchars, file=outname)
for taxa, chars in zip(taxa_list, chars_list):
    print(taxa, chars, sep="          ", file=outname)
