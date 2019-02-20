import glob, sys
from collections import defaultdict
from operator import itemgetter

prefix = sys.argv[1]

d = defaultdict(list)

for line in open("SAN_results/results.txt", "r"):
    arr = line.strip("\n").split(" ")
    gqd = float(arr[1])

    if prefix == "gold":
        family, _, setting = arr[0].split("_")
        log_file = "SAN_results/"+arr[0].replace("_SAN_",".").replace(".trees", "")+".log.txt"
        setting = setting.replace(".trees","").replace(".","_")

        for line in open(log_file, "r"):
            if "\t" in line:
                nr_gens = line.split("\t")[0]
            elif "seconds" in line:
                n_secs = float(line.split(" ")[1])

        d[family].append((setting, gqd, float(nr_gens), n_secs))

    else:
        method, family, setting = arr[0].split("-")
        log_file = "SAN_results/"+arr[0].replace("_",".").replace(".trees", "")+".log.txt"
        setting = setting.replace(".trees", "").split("_")[1]
        setting = setting.replace(".","_")

        if method == prefix:
            for line in open(log_file, "r"):
                if "\t" in line:
                    nr_gens = line.split("\t")[0]
                elif "seconds" in line:
                    n_secs = float(line.split(" ")[1])

            d[family].append((setting, gqd, float(nr_gens), n_secs))




fams = sorted(list(d.keys()))
top_k = 10

print("Family", "StepSize.InitTemp", "GQD", "Nr. Generations", "Nr. seconds", sep="\t")
for k in fams:
    vals = sorted(d[k], key=itemgetter(1,3))
    for val in vals[:top_k]:
        print(k, *val, sep="\t")
    print()

