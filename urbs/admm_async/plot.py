# TODO the whole thing

from os.path import join
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def index_lists(results: pd.DataFrame, n_clusters: int) -> List[List[int]]:
    # TODO: docstring

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
    }

    if centralized_objective is not None:
        series['obj_gap'] = objective_gap

    return series


def series_cutoff(series, lower, upper):
    new_series = {}
    for name, data in series.items():
        if name == 'avg_iter':
            new_series[name] = data
        new_series[name] = [min(upper, max(lower, x)) for x in data]
    return new_series


def beginning_of_the_end(series, threshold):
    """
    Return the index of the first value in `series` such that the value and all subsequent
    values are below `threshold`.
    """
    if series[-1] >= threshold:
        return -1

    for i, elem in enumerate(reversed(series)):
        if elem >= threshold:
            return len(series) - i # not -1 because we want the index *after* the element that we found

    return 0


def plot_results(
    result_dir: str,
    results: pd.DataFrame,
    metadata: Dict,
    centralized_objective=None,
    objective_tolerance=0.01,
    mode='combined',
    plot_rho=True,
    colors=None,
    ):
    """
    Plot the results and save them to `result_dir`.

    `results_dict` is a dict as returned by `.runfunctions_admm.run_regional`.
    """
    if mode not in ['combined', 'separate', 'both']:
        raise ValueError('Invalid mode')

    combined = mode in ['combined', 'both']
    separate = mode in ['separate', 'both']

    if colors is None:
        colors = {
            'primal': 'blue',
            'dual': 'orange',
            'mismatch': 'green',
            'obj': 'red',
            'rho': 'black',
        }

    admmopt = metadata['admmopt']
    n_clusters = metadata['n_clusters']
    series = data_series(results, n_clusters, centralized_objective)
    series = series_cutoff(series, 10**(-4), 10**8)

    if combined:
        fig_combined, ax_combined = plt.subplots()
        ax_combined.set_yscale('log')
        ax_combined.set_xlabel('avg local iterations')
        ax_combined.set_title('Results per Iteration')

        primal_convergence = series['avg_iter'][
            beginning_of_the_end(series['max_primal'], admmopt['primal_tolerance'])
        ]
        if primal_convergence >= 0:
            ax_combined.axvline(primal_convergence, color=colors['primal'])

        dual_convergence = series['avg_iter'][
            beginning_of_the_end(series['max_dual'], admmopt['dual_tolerance'])
        ]
        if dual_convergence >= 0:
            ax_combined.axvline(dual_convergence, color=colors['dual'])

        mismatch_convergence = series['avg_iter'][
            beginning_of_the_end(series['max_mismatch'], admmopt['mismatch_tolerance'])
        ]
        if mismatch_convergence >= 0:
            ax_combined.axvline(mismatch_convergence, color=colors['mismatch'])

    if 'max_primal' in series:
        if separate:
            fig, ax = fig_primal()
            ax.axhline(admmopt['primal_tolerance'], color='black', linestyle='dashed')
            ax.plot(series['avg_iter'], series['max_primal'])
            if plot_rho:
                ax.plot(series['avg_iter'], series['max_rho'], color=colors['rho'], label='rho')
            fig.savefig(join(result_dir, 'primal.svg'))
            plt.close(fig)
        if combined:
            ax_combined.plot(series['avg_iter'], series['max_primal'], label='primal gap', color=colors['primal'])

    if 'max_dual' in series:
        if separate:
            fig, ax = fig_dual()
            ax.axhline(admmopt['dual_tolerance'], color='black', linestyle='dashed')
            ax.plot(series['avg_iter'], series['max_dual'])
            if plot_rho:
                ax.plot(series['avg_iter'], series['max_rho'], color=colors['rho'], label='rho')
            fig.savefig(join(result_dir, 'dual.svg'))
            plt.close(fig)
        if combined:
            ax_combined.plot(series['avg_iter'], series['max_dual'], label='dual gap', color=colors['dual'])

    if 'max_mismatch' in series:
        if separate:
            fig, ax = fig_mismatch()
            ax.axhline(admmopt['mismatch_tolerance'], color='black', linestyle='dashed')
            ax.plot(series['avg_iter'], series['max_mismatch'])
            if plot_rho:
                ax.plot(series['avg_iter'], series['max_rho'], color=colors['rho'], label='rho')
            fig.savefig(join(result_dir, 'mismatch.svg'))
            plt.close(fig)
        if combined:
            ax_combined.plot(series['avg_iter'], series['max_mismatch'], label='max mismatch', color=colors['mismatch'])

    if 'obj_gap' in series:
        if separate:
            fig, ax = fig_objective()
            ax.axhline(0.01, color='black', linestyle='dashed')
            ax.plot(series['avg_iter'], series['obj_gap'])
            if plot_rho:
                ax.plot(series['avg_iter'], series['max_rho'], color=colors['rho'], label='rho')
            fig.savefig(join(result_dir, 'objective.svg'))
            plt.close(fig)
        if combined:
            ax_combined.plot(series['avg_iter'], series['obj_gap'], label='objective gap', color=colors['obj'])

            objective_convergence = series['avg_iter'][
                beginning_of_the_end(series['obj_gap'], objective_tolerance)
            ]
            if objective_convergence >= 0:
                ax_combined.axvline(objective_convergence, color=colors['obj'])

    if combined:
        if plot_rho:
            ax_combined.plot(series['avg_iter'], series['max_rho'], label='penalty', color=colors['rho'])
        ax_combined.legend()
        fig_combined.savefig(join(result_dir, 'combined.svg'))
        plt.close(fig_combined)


def fig_primal():
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('max primal gap')
    ax.set_title('Primal Gap per Iteration')
    return fig, ax


def fig_dual():
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('max dual gap')
    ax.set_title('Dual Gap per Iteration')
    return fig, ax


def fig_mismatch():
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('max constraint mismatch')
    ax.set_title('Constraint Mismatch per Iteration')
    return fig, ax


def fig_objective():
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('objective gap')
    ax.set_title('Objective Gap per Iteration')
    return fig, ax

def fig_raw_dual():
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('raw dual gap')
    ax.set_title('Raw Dual Gap per Iteration')
    return fig, ax
