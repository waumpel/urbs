# Asynchronous ADMM for the regional decomposition of energy system models

This folder contains a fork of the [urbs](https://github.com/tum-ens/urbs) software that I
developed over the course of the thesis.
My main contributions are the implementation of asynchronous ADMM and the scripts for the
experiments described in the thesis.
The former can be found in the `urbs/admm_async` folder, the latter in the root folder.

## Installation

Python 3.6 is recommended.
The dependencies are listed in `urbs-admm.yml`;
to install them with conda, use `conda env create -f urbs-admm.yml`,
then activate the environment with `conda activate urbs-admm`.
Further, the [Gurobi](https://www.gurobi.com/) solver is required.
An academic license can be requested on the website.

## Experiments

To accurately reproduce the results of the thesis, a machine with at least 35 processing
units and about 65GB of RAM is needed.

The input files for the Germany and Europe models are located in the `Input` folder.
The scripts for running the experiments are located in the root folder.
After running a script, the results can be found in the `result` folder.

To solve a model with the centralized approach, run the `run_germany.py` or `run_europe.py`
script with the desired number of timesteps:

    python run_germany.py -t=1

Once the problem is solved, the value of the objective function is printed to the terminal.
This value is needed for calculating the objective gap in the subsequent experiments.

The scripts for the experiments are numbered in the order in which they appear in the thesis,
from `01_ger1_base.py` to `11_eur1000.py`.
Simply run them using `python <script_name>`.
Runs that encountered numerical issues in the experiments are excluded as they cause the
script to get stuck.
After running a script, the corresponding result folder will contain a `plot.py` script that
can be used to create plots and compare the different runs of each experiment.
Before it can be used, the objective function value from the centralized run must be
entered in the line

    centralized_objective = None # REQUIRED

Running `plot.py` with no options only creates the `comparison.txt` file;
passing the options `-c` or `-d` also creates combined plots for all runs or detailed plots
for each individual run.
