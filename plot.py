import argparse
import io
import json
import os
from os.path import join, isdir
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import Inf


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


def words_in_line(line):
    if line.endswith('\n'):
        line = line[:-1]
    return line.split(' ')


def read_results(f: io.TextIOWrapper) -> pd.DataFrame:
    """
    Read `AdmmMetadata` objects persisted as strings from a file into a `DataFrame`.
    """
    lines = f.readlines()
    n_headers = len(words_in_line(lines[0]))
    skiprows = []
    for i, line in enumerate(lines):
        if line.endswith('\n'):
            line = line[:-1]
        words = line.split(' ')
        if len(words) != n_headers:
            skiprows.append(i)
            raise RuntimeWarning(f'Results file contains malformed line: line {i}')
    f.seek(0)
    df = pd.read_csv(
        f,
        sep=' ',
        header=0,
        skiprows=skiprows
    )
    return df


def index_lists(results: pd.DataFrame, n_clusters: int) -> List[List[int]]:
    process_ids = results['process_id'].to_list()
    # Find the first iteration where all clusters have returned a result.
    first_full_iteration = max(
        process_ids.index(i) for i in range(n_clusters) # first occurrence of each cluster
    )

    # For each cluster, find the index of the last result sent up to the first full iteration.
    split = first_full_iteration + 1
    head = process_ids[:split].copy()
    head.reverse()
    index_map = { i: len(head) - 1 - head.index(i) for i in range(n_clusters) }
    lists = [ list(index_map.values()) ]

    # Build the remaining index maps.
    tail = process_ids[split:]
    for i, ID in enumerate(tail):
        index_map[ID] = split + i
        lists.append(list(index_map.values()))

    return lists


def data_series(results: pd.DataFrame, n_clusters: int, centralized_objective=None):
    lists = index_lists(results, n_clusters)

    avg_iteration = []
    max_primalgap = []
    max_dualgap = []
    max_mismatch = []
    max_rho = []
    max_time = []
    if centralized_objective is not None:
        objective_gap = []

    for index_list in lists:
        data_slice = results.iloc[index_list] # select rows
        avg_iteration.append(sum(data_slice['local_iteration']) / n_clusters)
        max_primalgap.append(max(data_slice['primalgap']))
        max_dualgap.append(max(data_slice['dualgap']))
        max_mismatch.append(
            np.nan if np.nan in data_slice['mismatch'] else max(data_slice['mismatch'])
        )
        max_rho.append(max(data_slice['penalty']))
        max_time.append(max(data_slice['stop_time']))
        if centralized_objective is not None:
            objective_gap.append(
                abs(centralized_objective - sum(data_slice['objective'])) / centralized_objective
            )

    series = {
        'avg_iter': avg_iteration,
        'max_primal': max_primalgap,
        'max_dual': max_dualgap,
        'max_mismatch': max_mismatch,
        'max_rho': max_rho,
        'max_time': max_time,
    }

    if centralized_objective is not None:
        series['obj_gap'] = objective_gap

    return series


def max_gap(series, dualgap=False):
    gaps = [series['max_primal'], series['max_mismatch']]
    if dualgap:
        gaps.append(series['max_dual'])
    return [max(args) for args in zip(*gaps)]


def cutoff_filter(seq, lowcut, highcut):
    return [min(highcut, max(lowcut, x)) for x in seq]


if __name__ == '__main__':

    options = argparse.ArgumentParser()
    options.add_argument('-d', '--detailed', action='store_true')
    options.add_argument('-c', '--combined', action='store_true')

    args = options.parse_args()

    detailed = args.detailed
    combined = args.combined

    out_dir = '.'

    # Automatically find subdirs for plotting
    plot_dirs = [
        subdir for subdir in os.listdir()
        if (isdir(subdir)
            and ITER_RESULTS_FILE in os.listdir(subdir)
            and METADATA_FILE in os.listdir(subdir))
    ]
    # Or specify subdirs manually
    # plot_dirs = [
    #     '',
    # ]

    if len(plot_dirs) > len(COLORS):
        raise RuntimeError('Not enough colors')

    # Use directory names as labels
    labels = plot_dirs
    # Or specify labels manually
    # labels = [
    #     '',
    # ]

    title_suffix = ''

    centralized_objective = None

    # Options
    dualgap = False
    convergence_cutoff = (-Inf, Inf)
    objective_cutoff = (-Inf, Inf)
    details_cutoff = (-Inf, Inf)



    # === LET THE PLOTTING BEGIN ===

    if detailed:
        print('Creating detailed plots')
    if combined:
        print('Creating combined plots')
    if not (detailed or combined):
        print('Creating no plots, only runtime comparison')

    convergence_list = []

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 13}

    plt.rc('font', **font)

    if combined:
        fig_gaps, ax_gaps = plt.subplots()
        ax_gaps.set_yscale('log')
        ax_gaps.set_xlabel('time (seconds)')
        ax_gaps.set_ylabel(r'$\Gamma_{\max}$' if dualgap else r'$\Gamma_{p, c}$')
        ax_gaps.set_title('Convergence over time' + title_suffix)

        fig_obj, ax_obj = plt.subplots()
        ax_obj.set_yscale('log')
        ax_obj.set_xlabel('time (seconds)')
        ax_obj.set_ylabel(r'$\Theta$')
        ax_obj.set_title('Objective gap over time' + title_suffix)

    for subdir, label, color in zip(plot_dirs, labels, COLORS):
        print(f"{subdir} ...")

        with open(join(subdir, METADATA_FILE), 'r', encoding='utf8') as f:
            metadata = json.load(f)

        with open(join(subdir, ITER_RESULTS_FILE), 'r', encoding='utf8') as f:
            results = read_results(f)

        n_clusters = len(metadata['clusters'])
        series = data_series(results, n_clusters, centralized_objective)

        convergence = max_gap(series, dualgap)
        index = next((k for k, val in enumerate(convergence) if val < 0.01), -1)
        if index == -1:
            time = '-'
            obj_gap = '-'
        else:
            time = series['max_time'][index]
            obj_gap = series['obj_gap'][index]
            iter = series['avg_iter'][index]
        convergence_list.append((subdir, time, obj_gap, iter))

        if detailed:
            fig, ax = plt.subplots()
            ax.set_yscale('log')
            ax.set_xlabel('time (seconds)')
            ax.set_title(label)

            ax.plot(series['max_time'], cutoff_filter(series['max_primal'], *details_cutoff),
                    label=r'$\Gamma_p$')
            ax.plot(series['max_time'], cutoff_filter(series['max_dual'], *details_cutoff),
                    label=r'$\Gamma_d$')
            ax.plot(series['max_time'], cutoff_filter(series['max_mismatch'], *details_cutoff),
                    label=r'$\Gamma_c$')
            ax.plot(series['max_time'], cutoff_filter(series['obj_gap'], *details_cutoff),
                    label=r'$\Theta$')

            ax.legend()
            fig.savefig(join(out_dir, f'details-{subdir}.pdf'))
            plt.close(fig)

        if combined:
            ax_gaps.plot(series['max_time'],
                         cutoff_filter(convergence, *convergence_cutoff),
                         label=label, color=color)
            ax_obj.plot(series['max_time'],
                        cutoff_filter(series['obj_gap'], *objective_cutoff),
                        label=label, color=color)

    print(f'Saving to "{out_dir}" ...')

    if combined:
        ax_gaps.legend()
        fig_gaps.savefig(join(out_dir, 'convergence.pdf'))
        plt.close(fig_gaps)

        ax_obj.legend()
        fig_obj.savefig(join(out_dir, 'objective.pdf'))
        plt.close(fig_obj)

    with open(join(out_dir, 'comparison.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(' '.join(str(x) for x in l) for l in convergence_list))
