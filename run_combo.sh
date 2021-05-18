#!/bin/bash
# Run ANN training; info flow analysis; tradeoff analysis; scaling analysis
# All parallelization requires 'sem' from GNU parallel: sudo apt install parallel

dataset="adult"
results_dir="results-$dataset"
mkdir -p "$results_dir"

# Create outfiles to record start and end times and outputs of processes
outfile_root="$results_dir/run-$(date '+%Y%m%d%H%M%S')"
outfile="$outfile_root.out"
echo -n "Start: " >"$outfile"
echo $(date) >>"$outfile"

retrain_flag=0    # Retrain and reanalyze if set
reanalyze_flag=0  # Reanalyze if set
runs=10           # How many trials to run

# Train the ANNs (not parallelized)
if [ $retrain_flag == 1 ]; then
	echo "Training ANNs"
	python3 -u nn.py -d $dataset --runs $runs | tee "$outfile_root-train.out"
fi

# Run info flow analysis in parallel
if [ $retrain_flag == 1 ] || [ $reanalyze_flag == 1 ]; then  # Retrain implies reanalyze
	export MKL_NUM_THREADS=1
	echo "Analyzing info flow in ANNs"
	for (( j=0 ; j<$runs ; j=j+1 )); do
		echo -n "$j "
		sem -j 8 --id analyze "python3 -u analyze_info_flow.py -d $dataset --runs $runs -j $j > $outfile_root-analyze-$j.out"
	done
	echo
	echo "Waiting for jobs to complete..."
	sem --wait --id analyze
	python3 -u analyze_info_flow.py -d $dataset --runs $runs --concatenate
fi

# Run tradeoff analysis in parallel
export MKL_NUM_THREADS=1
i=0
while IFS='-' read -a params; do  # Read and split params from file
	echo -n "$i "
	metric=${params[0]}
	method=${params[1]}
	pruneamt=${params[2]}
	sem -j 8 --id tradeoff "python3 -u tradeoff_analysis.py -d $dataset --metric $metric --method $method --pruneamt $pruneamt --runs $runs >$outfile_root-tradeoff-$i.out"
	let i=i+1
done < "$results_dir/combos.txt"  # Input file for the while loop
echo
sem --wait --id tradeoff

# Record end time
echo -n "Finished: " >>"$outfile"
echo $(date) >>"$outfile"
