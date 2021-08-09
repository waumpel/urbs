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

args = options.parse_args()

RESULTS_FILE = 'iteration_results.txt'
METADATA_FILE = 'metadata.json'

result_dir = join('result', args.dir)
contents = os.listdir(result_dir)
plot_dirs = []
if RESULTS_FILE in contents:
    plot_dirs.append(result_dir)

for d in contents:
    path = join(result_dir, d)
    if (isdir(path)
        and RESULTS_FILE in os.listdir(path)
        and METADATA_FILE in os.listdir(path)):
        plot_dirs.append(path)

for d in plot_dirs:
    with open(join(d, METADATA_FILE), 'r', encoding='utf8') as f:
        metadata = json.load(f)
    with open(join(d, RESULTS_FILE), 'r', encoding='utf8') as f:
        results = read_results(f)

    plot.plot_results(
        d,
        results,
        metadata,
        centralized_objective=args.centralized,
        plot_rho=True,
    )
