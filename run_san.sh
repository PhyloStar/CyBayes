#!/bin/bash

for t in 1 5 10 20 40 80 100
do
    for j in 10 20 30 40 50 60 70 80 90 100
    do
#        python3 san.py -i pruned_dataset/data-aa-58-200.paps.phy -m F81 -t $t -d bin -o SAN_results/AA_SAN_$t.$j -t0 $j > SAN_results/AA.$t.$j.log.txt
#        python3 san.py -i pruned_dataset/data-an-45-210.paps.phy -m F81 -t $t -d bin -o SAN_results/An_SAN_$t.$j -t0 $j > SAN_results/An.$t.$j.log.txt
#        python3 san.py -i pruned_dataset/data-ie-42-208.paps.phy -m F81 -t $t -d bin -o SAN_results/IE_SAN_$t.$j -t0 $j > SAN_results/IE.$t.$j.log.txt&
#        python3 san.py -i pruned_dataset/data-st-64-110.paps.phy -m F81 -t $t -d bin -o SAN_results/ST_SAN_$t.$j -t0 $j > SAN_results/ST.$t.$j.log.txt
#        python3 san.py -i pruned_dataset/data-pn-67-183.paps.phy -m F81 -t $t -d bin -o SAN_results/PN_SAN_$t.$j -t0 $j > SAN_results/PN.$t.$j.log.txt&
#        wait
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/aa-58-200.glot_cleaned.tre SAN_results/AA_SAN_$t.$j.trees >> SAN_results/results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/an-45-210.glot_cleaned.tre SAN_results/An_SAN_$t.$j.trees >> SAN_results/results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/ie-42-208.glot_cleaned.tre SAN_results/IE_SAN_$t.$j.trees >> SAN_results/results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/st-64-110.glot_cleaned.tre SAN_results/ST_SAN_$t.$j.trees >> SAN_results/results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/pn-67-183.glot_cleaned.tre SAN_results/PN_SAN_$t.$j.trees >> SAN_results/results.txt
    done
done
