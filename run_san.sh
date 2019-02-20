#!/bin/bash

for t in 1 5 10 20 40 80 100
do
    for j in 10 20 30 40 50 60 70 80 90 100
    do
#        python3 san.py -i pruned_dataset/data-aa-58-200.paps.phy -m F81 -t $t -d bin -o SAN_results/AA_SAN_$t.$j -t0 $j > SAN_results/AA.$t.$j.log.txt
#        python3 san.py -i pruned_dataset/data-an-45-210.paps.phy -m F81 -t $t -d bin -o SAN_results/An_SAN_$t.$j -t0 $j > SAN_results/An.$t.$j.log.txt

#        python3 san.py -i pruned_dataset/data-st-64-110.paps.phy -m F81 -t $t -d bin -o SAN_results/ST_SAN_$t.$j -t0 $j > SAN_results/ST.$t.$j.log.txt
#        python3 san.py -i pruned_dataset/data-pn-67-183.paps.phy -m F81 -t $t -d bin -o SAN_results/PN_SAN_$t.$j -t0 $j > SAN_results/PN.$t.$j.log.txt&

#        python3 san.py -i fcd_glotto_phy/cc-aa-58-200.phy -m F81 -t $t -d bin -o SAN_results/cc-aa-san_$t.$j -t0 $j > SAN_results/cc-aa-san.$t.$j.log.txt
#        python3 san.py -i fcd_glotto_phy/infomap-aa-58-200.phy -m F81 -t $t -d bin -o SAN_results/infomap-aa-san_$t.$j -t0 $j > SAN_results/infomap-aa-san.$t.$j.log.txt&

#        python3 san.py -i fcd_glotto_phy/cc-an-45-210.phy -m F81 -t $t -d bin -o SAN_results/cc-an-san_$t.$j -t0 $j > SAN_results/cc-an-san.$t.$j.log.txt
#        python3 san.py -i fcd_glotto_phy/infomap-an-45-210.phy -m F81 -t $t -d bin -o SAN_results/infomap-an-san_$t.$j -t0 $j > SAN_results/infomap-an-san.$t.$j.log.txt&

#        python3 san.py -i fcd_glotto_phy/cc-ie-42-208.phy -m F81 -t $t -d bin -o SAN_results/cc-ie-san_$t.$j -t0 $j > SAN_results/cc-ie-san.$t.$j.log.txt
#        python3 san.py -i fcd_glotto_phy/infomap-ie-42-208.phy -m F81 -t $t -d bin -o SAN_results/infomap-ie-san_$t.$j -t0 $j > SAN_results/infomap-ie-san.$t.$j.log.txt&

#        python3 san.py -i fcd_glotto_phy/cc-pn-67-183.phy -m F81 -t $t -d bin -o SAN_results/cc-pn-san_$t.$j -t0 $j > SAN_results/cc-pn-san.$t.$j.log.txt
#        python3 san.py -i fcd_glotto_phy/infomap-pn-67-183.phy -m F81 -t $t -d bin -o SAN_results/infomap-pn-san_$t.$j -t0 $j > SAN_results/infomap-pn-san.$t.$j.log.txt&

#        python3 san.py -i fcd_glotto_phy/cc-st-64-110.phy -m F81 -t $t -d bin -o SAN_results/cc-st-san_$t.$j -t0 $j > SAN_results/cc-st-san.$t.$j.log.txt
#        python3 san.py -i fcd_glotto_phy/infomap-st-64-110.phy -m F81 -t $t -d bin -o SAN_results/infomap-st-san_$t.$j -t0 $j > SAN_results/infomap-st-san.$t.$j.log.txt&


#        wait

        python3 gqd1.py ~/AutoCogPhylo/gold_trees/aa-58-200.glot_cleaned.tre SAN_results/cc-aa-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/aa-58-200.glot_cleaned.tre SAN_results/infomap-aa-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt

        python3 gqd1.py ~/AutoCogPhylo/gold_trees/an-45-210.glot_cleaned.tre SAN_results/cc-an-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/an-45-210.glot_cleaned.tre SAN_results/infomap-an-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt

        python3 gqd1.py ~/AutoCogPhylo/gold_trees/ie-42-208.glot_cleaned.tre SAN_results/cc-ie-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/ie-42-208.glot_cleaned.tre SAN_results/infomap-ie-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt

        python3 gqd1.py ~/AutoCogPhylo/gold_trees/st-64-110.glot_cleaned.tre SAN_results/cc-st-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/st-64-110.glot_cleaned.tre SAN_results/infomap-st-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt

        python3 gqd1.py ~/AutoCogPhylo/gold_trees/pn-67-183.glot_cleaned.tre SAN_results/cc-pn-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt
        python3 gqd1.py ~/AutoCogPhylo/gold_trees/pn-67-183.glot_cleaned.tre SAN_results/infomap-pn-san_$t.$j.trees >> SAN_results/cc_infomap_SAN_results.txt



#        python3 gqd1.py ~/AutoCogPhylo/gold_trees/aa-58-200.glot_cleaned.tre SAN_results/AA_SAN_$t.$j.trees >> SAN_results/results.txt
#        python3 gqd1.py ~/AutoCogPhylo/gold_trees/an-45-210.glot_cleaned.tre SAN_results/An_SAN_$t.$j.trees >> SAN_results/results.txt
#        python3 gqd1.py ~/AutoCogPhylo/gold_trees/ie-42-208.glot_cleaned.tre SAN_results/cc-ie-san_$t.$j.trees >> SAN_results/results.txt
#        python3 gqd1.py ~/AutoCogPhylo/gold_trees/ie-42-208.glot_cleaned.tre SAN_results/infomap-ie-san_$t.$j.trees >> SAN_results/results.txt
#        python3 gqd1.py ~/AutoCogPhylo/gold_trees/st-64-110.glot_cleaned.tre SAN_results/ST_SAN_$t.$j.trees >> SAN_results/results.txt
#        python3 gqd1.py ~/AutoCogPhylo/gold_trees/pn-67-183.glot_cleaned.tre SAN_results/PN_SAN_$t.$j.trees >> SAN_results/results.txt
    done
done
