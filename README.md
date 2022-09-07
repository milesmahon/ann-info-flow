# Can Information Flows Suggest Targets for Interventions in Neural Circuits?

Source code and instructions for the paper.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

You will also need to separately install [GNU Parallel](https://www.gnu.org/software/parallel/) in order to execute `run_combos.sh`.

Additionally, create the following symlinks for basic functionality:

```
ln -s nn_small.py nn.py
ln -s data_utils_small.py data_utils.py
ln -s results-adult-small results-adult
```

## Description of Files

The main scripts are:
- `param_utils.py`: Contains most parameter settings used throughout all scripts
- `data_utils.py`: Contains code for generating and extracting the datasets
- `nn.py`: Contains the ANN model and training code
- `analyze_info_flow.py`: Considers a trained neural network and analyzes the information flows on the network using test data
- `tradeoff_analysis.py`: Makes use of the information flows analyzed by `analyze_info_flow.py` and attempts to use different pruning strategies (encoded in `combos.txt`) to evaluate bias-accuracy tradeoffs
- `scaling_analysis.py`: Makes use of the information flows analyzed by `analyze_info_flow.py` and evaluates dependence between information flow and effect of full-pruning for each edge
- `run_combo.sh`: Bash script which runs analyses as necessary, parallelizing to the extent possible.
- `info_measures.py`: Utility script containing methods for estimating information measures
- `pruning.py`: Utility script containing methods for pruning an ANN
- `utils.py`: Miscellaneous utility functions
- `plot_tradeoff.py`: Script used for plotting tradeoff results
- `plot_scaling.py`: Script used for plotting scaling results
- `combos.txt`: A text file containing a list of pruning combinations used by `tradeoff_analysis.py` and `plot_tradeoff.py`
- `results-<dataset>`: Directories containing output data files used in the paper. `results-adult` is currently a symbolic link that points to `results-adult-small`. If you wish to plot results of the modified Adult dataset trained on the larger ANN, please replace the link to point to `results-adult-large`. The same holds true before rerunning any of the code for the larger ANN.

`nn.py` and `data_utils.py` do not exist at the start. There are two such files, one each for the small and the large ANN. These are created when you run `run_combo.sh` (see below).

## Running the code

The best way to re-run all analyses is to use `run_combo.sh`.

Start by setting `dataset` and `info_meth` in `run_combo.sh`:

```
dataset="tinyscm"       # "tinyscm" represents the synthetic dataset. Change to "adult" for the modified Adult dataset
info_meth="kernel-svm"  # Change to "linear-svm" or "corr" for estimating information flow using linear-SVM or the Correlation-based approximation respectively
network="small"         # Change to "large" for running the Adult dataset analysis on the larger ANN
```

Then, select which analysis needs to be run by setting the respective flags to 1. For example, the following settings will train, and analyze information flow, for 100 different ANN weight initializations, but will not run the tradeoff or scaling analyses.

```
retrain_flag=1    # Retrain if set
reanalyze_flag=1  # Reanalyze if set
run_tradeoff=0    # Run tradeoff if set
run_scaling=0     # Run scaling if set
runs=100          # How many trials to run
```

Use the `num_parallel` parameter to set the maximum number of jobs that can be run in parallel.

```
num_parallel=8    # Maximum number of jobs that can be run in parallel
```

After setting parameter values, run the script with no arguments:

```
./run_combo.sh
```

Note that running this script this will overwrite results in the respective `results-<dataset>` folder.

## To regenerate the synthetic dataset

Set `force_regenerate` to True in `param_utils.py`. The synthetic dataset is stored in `results-tinyscm/data-10000.pkl`.

## Setting pruning combinations

To set which pruning strategies to execute when using `tradeoff_analysis.py`, or which pruning strategies to plot when using `plot_tradeoff.py`, you will need to edit `results-<dataset>/combos.txt`. This file is present in `./combos.txt` and is automatically copied over to `results-<dataset>/combos.txt` when you run `run_combo.sh`. This is a text file that has a very limited syntax, and is parsed to determine pruning combinations.

For example, `combos.txt` may contain:

```
biasacc-node-1
biasacc-edge-2
accbias-edge-4
accbias-path-1
```

This combos.txt file will run (or plot) the following four different pruning strategies:
1. Metric: weighted bias-to-accuracy flow ratio; method: node; level: 1 node pruned
1. Metric: weighted bias-to-accuracy flow ratio; method: edge; level: 2 edges pruned
1. Metric: weighted accuracy-to-bias flow ratio; method: edge; level: 4 edges pruned
1. Metric: weighted accuracy-to-bias flow ratio; method: path; level: 1 path pruned

The metric can be either `biasacc` or `accbias`, and the pruning method can be one of `node`, `edge`, or `path` (refer Section 3.2 in the main paper). The level must be a small integer, fewer than the total number of nodes/edges/paths in the ANN (see legends in the tradeoff plots for examples).

Note that we have provided exemplar combos.txt files under both results-tinyscm and results-adult. These are copies of `combos_nodes.txt`, which produce results for the node-pruning method. We have also provided examples of combos.txt for the edge-pruning and path-pruning methods under the respective `results-<dataset>` directory.

## Plotting figures

These scripts should work on the provided result data. But if re-running all analyses, please ensure the respective analysis has completed before attempting to plot.

For plotting tradeoff figures, set `results-<dataset>/combos.txt` as described above, and run

```
./plot_tradeoff.py <dataset> <info_meth>
```

Colors will _not_ be replicated as in the paper.

For plotting scaling figures, simply run

```
./plot_scaling.py <dataset> <info_meth>
```

For plotting the ANN visualizations (shown in the appendix), run

```
./plot_utils.py <dataset> <info_meth> <run_number>
```

The run number is an integer between 0 and `$runs` (which was set in `run_combo.sh`).
