import argparse
import json
import os
from os.path import join, isdir

from urbs.admm_async import read_results
import urbs.admm_async.plot as plot

options = argparse.ArgumentParser()
options.add_argument('-d', '--dir', required=True,
                     help='Result directory for which plots should be generated')
options.add_argument('-c', '--centralized', type=float, default=None,
                     help='Objective value of the centralized solution')
options.add_argument('-t', '--tolerance', type=float, default=None,
                     help='Tolerance threshold for primal gap, dual gap, and constraint mismatch')

args = options.parse_args()

ITER_RESULTS_FILE = 'iteration_results.txt'
METADATA_FILE = 'metadata.json'

result_dir = join('result', args.dir)
contents = os.listdir(result_dir)

for subdir in contents:
    subdir_path = join(result_dir, subdir)
    if (isdir(subdir_path)
        and ITER_RESULTS_FILE in os.listdir(subdir_path)
        and METADATA_FILE in os.listdir(subdir_path)):

        with open(join(subdir_path, METADATA_FILE), 'r', encoding='utf8') as f:
            metadata = json.load(f)

        with open(join(subdir_path, ITER_RESULTS_FILE), 'r', encoding='utf8') as f:
            iter_results = read_results(f)

        plot.plot_results(
            subdir_path,
            iter_results,
            metadata,
            centralized_objective=args.centralized,
            primal_tolerance=args.tolerance,
            dual_tolerance=args.tolerance,
            mismatch_tolerance=args.tolerance,
            plot_rho=True,
        )
