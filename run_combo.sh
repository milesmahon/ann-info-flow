#!/bin/bash
# Run ANN training; info flow analysis; tradeoff analysis; scaling analysis
# All parallelization requires 'sem' from GNU parallel: sudo apt install parallel

dataset="adult"
results_dir="results-$dataset"
mkdir -p "$results_dir"

# Create outfiles to record start and end times and outputs of processes
outfile_root="$results_dir/run-$(date '+%Y%m%dT%H%M')"
outfile="$outfile_root.out"
echo -n "Start: " >"$outfile"
echo $(date) >>"$outfile"

export MKL_NUM_THREADS=1

retrain_flag=0    # Retrain if set
reanalyze_flag=1  # Reanalyze if set
run_tradeoff=0    # Run tradeoff if set
run_scaling=0     # Run scaling if set
runs=10           # How many trials to run

info_meth="corr"

# Train the ANNs (not parallelized)
if [ $retrain_flag == 1 ]; then
	echo "Training ANNs"
	python3 -u nn.py -d $dataset --runs $runs | tee "$outfile_root-train.out"
fi

# Run info flow analysis, parallelized over runs
if [ $reanalyze_flag == 1 ]; then
	echo "Analyzing info flow in ANNs"
	for (( j=0 ; j<$runs ; j=j+1 )); do
		echo -n "$j "
		sem -j 8 --id analyze "python3 -u analyze_info_flow.py -d $dataset --info-method $info_meth --subfolder $info_meth --runs $runs -j $j > $outfile_root-analyze-$j.out"
	done
	echo
	echo "Waiting for jobs to complete..."
	sem --wait --id analyze
	python3 -u analyze_info_flow.py -d $dataset --subfolder $info_meth --runs $runs --concatenate
fi

# Run tradeoff analysis, parallelized over combos
if [ $run_tradeoff == 1 ]; then
	i=0
	while IFS='-' read -a params; do  # Read and split params from file
		echo -n "$i "
		metric=${params[0]}
		method=${params[1]}
		pruneamt=${params[2]}
		sem -j 8 --id tradeoff "python3 -u tradeoff_analysis.py -d $dataset --info-method $info_meth --subfolder $info_meth --metric $metric --method $method --pruneamt $pruneamt --runs $runs >$outfile_root-tradeoff-$i.out"
		let i=i+1
	done < "$results_dir/combos.txt"  # Input file for the while loop
	echo
	sem --wait --id tradeoff
fi

# Run scaling analysis, parallelized over runs
if [ $run_scaling == 1 ]; then
	echo "Running scaling analysis"
	for (( j=0 ; j<$runs ; j=j+1 )); do
		echo -n "$j "
		sem -j 8 --id scaling "python3 -u scaling_analysis.py -d $dataset --info-method $info_meth --subfolder $info_meth --runs $runs -j $j > $outfile_root-analyze-$j.out"
	done
	echo
	echo "Waiting for jobs to complete..."
	sem --wait --id scaling
	python3 -u scaling_analysis.py -d $dataset --subfolder $info_meth --runs $runs --concatenate
fi

# Record end time
echo -n "Finished: " >>"$outfile"
echo $(date) >>"$outfile"
