import argparse
import json
import os
from os.path import join, isdir
from typing import List, Tuple

import matplotlib.pyplot as plt

from urbs.admm_async import read_results
import urbs.admm_async.plot as plot


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
    'b',
    'm',
]


def find_plotdirs(root: str) -> List[Tuple[str, str]]:
    contents = os.listdir(root)

    plot_dirs = []
    for subdir in contents:
        subdir_path = join(root, subdir)
        if isdir(subdir_path):
            subdir_contents = os.listdir(subdir_path)
            if (RESULT_FILE in subdir_contents
                and ITER_RESULTS_FILE in subdir_contents
                and METADATA_FILE in subdir_contents):
                plot_dirs.append((subdir, subdir_path))

    return plot_dirs


if __name__ == '__main__':

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

    result_dir = join('result', args.dir)
    contents = os.listdir(result_dir)

    # Automatically find subdirs for plotting
    plot_dirs = find_plotdirs(result_dir)

    # Or specify subdirs manually
    # plot_dirs = [
    #     (subdir, join(result_dir, subdir)) for subdir in [
    #         '100',
    #         '1000',
    #         '10000',
    #         '100000',
    #     ]
    # ]

    if len(plot_dirs) > len(COLORS):
        raise RuntimeError('Not enough colors')

    fig_gaps, ax_gaps = plt.subplots()
    # fig_gaps, ax_gaps = plt.subplots(figsize=(12.8, 9.6))
    ax_gaps.set_yscale('log')
    ax_gaps.set_xlabel('time')
    ax_gaps.set_title('Max primal gap/constraint mismatch over time')
    ax_gaps.axhline(args.tolerance, color='black', linestyle='dashed')

    if args.centralized:
        fig_obj, ax_obj = plt.subplots()
        # fig_obj, ax_obj = plt.subplots(figsize=(12.8, 9.6))
        ax_obj.set_yscale('log')
        ax_obj.set_xlabel('time')
        ax_obj.set_title('Objective gap over time')
        ax_obj.axhline(args.obj_tol, color='black', linestyle='dashed')

    for i, (subdir, subdir_path) in enumerate(plot_dirs):
        print(f"Plotting {subdir}...")
        color = COLORS[i]

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

    if args.centralized:
        ax_obj.legend()
        fig_obj.savefig(join(result_dir, 'obj.svg'))
        plt.close(fig_obj)
