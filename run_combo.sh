#!/bin/bash

dataset="adult"
results_dir="results-$dataset"
mkdir -p "$results_dir"

# Create outfiles to record start and end times and outputs of processes
outfile_root="$results_dir/run-$(date '+%Y%m%d%H%M%S')"
outfile="$outfile_root.out"
echo -n "Start: " >"$outfile"
echo $(date) >>"$outfile"

retrain_flag=0    # Retrain and reanalyze if set
reanalyze_flag=1  # Reanalyze if set
runs=10           # How many trials to run

## Run analyze_info_flow.py for a combination of settings
##methods=("edge")
##metrics=("biasacc")
##pruneamts=("6")
#methods=("node" "edge")
#metrics=("biasacc" "accbias")
#pruneamts=("1" "2")
#rm -f "$results_dir/combos-tmp.txt"
#for method in "${methods[@]}"; do
#    for metric in "${metrics[@]}"; do
#        for num in "${pruneamts[@]}"; do
#            if [ $retrain_flag == 1 ]; then
#                ./analyze_info_flow.py -d $dataset --metric $metric --method $method --pruneamt $num --runs $runs --retrain --reanalyze
#                retrain_flag=0
#            else
#                ./analyze_info_flow.py -d $dataset --metric $metric --method $method --pruneamt $num --runs $runs
#            fi
#            echo "${metric}-${method}-${num}" >> "$results_dir/combos-tmp.txt"
#        done
#    done
#done
## Overwrite the older combos file - doing this separately prevents overwriting in case the script was interrupted midway
#mv "$results_dir/combos-tmp.txt" "$results_dir/combos.txt"

# TODO: Need to have a way of parallelizing over runs, not only over parameter combinations
# This needs to be fully done in advance of all parallel runs of pruning
if [ $retrain_flag == 1 ] || [ $reanalyze_flag == 1 ]; then
	# Retrain and reanalyze if retrain_flag is set; otherwise just reanalyze
	if [ $retrain_flag == 1 ]; then retrain_arg="--retrain"; else retrain_arg=""; fi
	python3 -u analyze_info_flow.py -d $dataset --runs $runs $retrain_arg --reanalyze --analyze-only | tee "$outfile_root-train.out"  # Should not have quotes around retrain arg here!
fi

# Run analyze_info_flow.py in parallel (requires 'sem' from GNU parallel: sudo apt install parallel)
export MKL_NUM_THREADS=1
i=0
while IFS='-' read -a params; do  # Read and split params from file
	echo -n "$i "
	metric=${params[0]}
	method=${params[1]}
	pruneamt=${params[2]}
	sem -j 8 "python3 -u analyze_info_flow.py -d $dataset --metric $metric --method $method --pruneamt $pruneamt --runs $runs >$outfile_root-$i.out"
	let i=i+1
done < "$results_dir/combos.txt"  # Input file for the while loop
echo
sem --wait

# Record end time
echo -n "Finished: " >>"$outfile"
echo $(date) >>"$outfile"
