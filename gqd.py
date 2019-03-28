import itertools as it
import sys, subprocess, glob
import numpy as np
from scipy import stats


gold_tree = sys.argv[1]
target_trees = sys.argv[2]

cutoff = float(sys.argv[3])

gqd = []
trees = [x.split("\t")[1] for x in open(target_trees, "r")]
 
for t1 in trees[int(cutoff*len(trees)):]:
    temp1 = open("temp1.txt", "w")
    temp1.write(t1.replace("_","").replace("-",""))
    temp1.close()
    a = subprocess.check_output(["./qdist", "temp1.txt", gold_tree])
    x=str(a).split("\\n")[1].split("\\t")
    gqd.append(float(x[4])/float(x[2]))
gqd = np.array(gqd)
print("Dataset|Mean|Stdev|Minimum|Maximum|No. of Trees|Original Trees")
print("------|------|------|------|------|------|------")
print(target_trees, np.round(np.mean(gqd),4), np.round(np.std(gqd),4), round(np.min(gqd),4), round(np.max(gqd), 4), gqd.shape[0], len(trees), sep="|")
#print("GQD ",gold_tree, target_trees, "Mean = ", np.round(np.mean(gqd),4), "Stdev = ",np.round(np.std(gqd),4), "Minimum = ", round(np.min(gqd),4), "at ", np.argmin(gqd), "Maximum = ", round(np.max(gqd), 4), " at ", np.argmax(gqd) ,"#. trees = ",gqd.shape[0], "#. original trees = ", len(trees))

#for i, g in enumerate(gqd):
#    print(i, g)
