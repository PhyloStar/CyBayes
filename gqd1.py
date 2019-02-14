import itertools as it
import sys, subprocess, glob
import numpy as np
from scipy import stats

cutoff = 0.0

gold_tree = sys.argv[1]
target_trees = sys.argv[2]

gqd = []
trees = [x.split("\t")[1] for x in open(target_trees, "r")]
 
for t1 in trees[int(cutoff*len(trees)):]:
    temp1 = open("temp1.txt", "w")
    temp1.write(t1.replace("_","").replace("-",""))
    temp1.close()
    a = subprocess.check_output(["./qdist", "temp1.txt", gold_tree])
    x=str(a).split("\\n")[1].split("\\t")
    gqd.append(float(x[4])/float(x[2]))

print(target_trees.split("/")[-1], gqd[-1])

#gqd = np.array(gqd)
#print(np.round(np.mean(gqd),4), np.round(np.std(gqd),4), gqd.shape[0])

