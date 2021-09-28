import argparse
import json
import os
from os.path import join, isdir

import matplotlib.pyplot as plt

from urbs.admm_async import read_results
import urbs.admm_async.plot as plot

options = argparse.ArgumentParser()
options.add_argument('-d', '--dir', required=True,
                     help='Result directory for which plots should be generated')
options.add_argument('-c', '--centralized', type=float, default=None,
                     help='Objective value of the centralized solution')
options.add_argument('-t', '--tolerance', type=float, default=0.01,
                     help='Tolerance for primal gap and constraint mismatch')
options.add_argument('-o', '--obj_tol', type=float, default=0.01,
                     help='Tolerance for objective gap')

args = options.parse_args()

RESULT_FILE = 'result.json'
ITER_RESULTS_FILE = 'iteration_results.txt'
METADATA_FILE = 'metadata.json'

COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]

result_dir = join('result', args.dir)
contents = os.listdir(result_dir)

fig_gaps, ax_gaps = plt.subplots()
ax_gaps.set_yscale('log')
ax_gaps.set_xlabel('time')
ax_gaps.set_title('Max primal gap/constraint mismatch over time')
ax_gaps.axhline(args.tolerance, color='black', linestyle='dashed')

if args.centralized:
    fig_obj, ax_obj = plt.subplots()
    ax_obj.set_yscale('log')
    ax_obj.set_xlabel('time')
    ax_obj.set_title('Objective gap over time')
    ax_obj.axhline(args.obj_tol, color='black', linestyle='dashed')

for subdir in contents:
    subdir_path = join(result_dir, subdir)
    if (isdir(subdir_path)
        and ITER_RESULTS_FILE in os.listdir(subdir_path)
        and METADATA_FILE in os.listdir(subdir_path)):

        if not COLORS:
            raise RuntimeError('Ran out of colors')
        color = COLORS.pop(0)

        with open(join(subdir_path, METADATA_FILE), 'r', encoding='utf8') as f:
            metadata = json.load(f)

        with open(join(subdir_path, ITER_RESULTS_FILE), 'r', encoding='utf8') as f:
            iter_results = read_results(f)

        with open(join(subdir_path, RESULT_FILE), 'r', encoding='utf8') as f:
            result = json.load(f)

        plot.plot_gaps(
            ax_gaps,
            color,
            subdir,
            iter_results,
            metadata,
            args.tolerance,
            args.tolerance,
        )

        if args.centralized:
            plot.plot_objective(
                ax_obj,
                color,
                subdir,
                iter_results,
                metadata,
                args.centralized,
                args.obj_tol,
            )

ax_gaps.legend()
fig_gaps.savefig(join(result_dir, 'gap.svg'))
plt.close(fig_gaps)

ax_obj.legend()
fig_obj.savefig(join(result_dir, 'obj.svg'))
plt.close(fig_obj)
