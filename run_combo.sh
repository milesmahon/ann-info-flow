#!/bin/bash
# Run ANN training; info flow analysis; tradeoff analysis; scaling analysis
# All parallelization requires 'sem' from GNU parallel: sudo apt install parallel
# The first time you use sem, run `parallel --citation` in a terminal to
# acknowledge and silence the citation notice.

dataset="adult-small"
info_meth="corr"

retrain_flag=0    # Retrain if set
reanalyze_flag=0  # Reanalyze if set
run_tradeoff=1    # Run tradeoff if set
run_scaling=1     # Run scaling if set
runs=1            # How many trials to run

num_parallel=1    # Maximum number of jobs that can be run in parallel

export MKL_NUM_THREADS=1

if [ $dataset == "mnist" ]; then
	ln -sf nn_cnn.py nn.py
	ln -sf data_utils_cnn.py data_utils.py
else
	ln -sf nn_large.py nn.py
	ln -sf data_utils_large.py data_utils.py
fi

results_dir="results-$dataset"
mkdir -p "$results_dir"
#cp combos.txt $results_dir

if [ $dataset == "tinyscm" ]; then
	cp data-10000.pkl $results_dir
fi

if [ $dataset == "mnist" ]; then
	mkdir $results_dir/filters
fi

# Create outfiles to record start and end times and outputs of processes
outfile_root="$results_dir/run-$(date '+%Y%m%dT%H%M')"
outfile="$outfile_root.out"
echo -n "Start: " >"$outfile"
echo $(date) >>"$outfile"

# Train the ANNs (not parallelized)
if [ $retrain_flag == 1 ]; then
	echo "Training ANNs"
	python3 -u train_ann.py -d $dataset --runs $runs | tee "$outfile_root-train.out"
fi

# Run info flow analysis, parallelized over runs
if [ $reanalyze_flag == 1 ]; then
	echo "Analyzing info flow in ANNs"
	for (( j=0 ; j<$runs ; j=j+1 )); do
		echo -n "$j "
		sem -j $num_parallel --id analyze "python3 -u analyze_info_flow.py -d $dataset --info-method $info_meth --subfolder $info_meth --runs $runs -j $j > $outfile_root-analyze-$j.out"
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
		sem -j $num_parallel --id tradeoff "python3 -u tradeoff_analysis.py -d $dataset --info-method $info_meth --subfolder $info_meth --metric $metric --method $method --pruneamt $pruneamt --runs $runs >$outfile_root-tradeoff-$i.out"
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
		sem -j $num_parallel --id scaling "python3 -u scaling_analysis.py -d $dataset --info-method $info_meth --subfolder $info_meth --runs $runs -j $j > $outfile_root-scaling-$j.out"
	done
	echo
	echo "Waiting for jobs to complete..."
	sem --wait --id scaling
	python3 -u scaling_analysis.py -d $dataset --subfolder $info_meth --runs $runs --concatenate
fi

# Record end time
echo -n "Finished: " >>"$outfile"
echo $(date) >>"$outfile"
