from os.path import join
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    Return -1 if there is no such index.
    """
    if series[-1] >= threshold:
        return -1

    for i, elem in enumerate(reversed(series)):
        if elem >= threshold:
            return len(series) - i # not -1 because we want the index *after* the element that we found

    return 0


def solver_times_dict(results: pd.DataFrame, n_clusters: int):
    solver_times = { k: [] for k in range(n_clusters) }
    for _, row in results.iterrows():
        ID = row['process_id']
        solver_time = row['stop_time'] - row['start_time']
        solver_times[ID].append(solver_time)

    return solver_times


def plot_results(
    result_dir: str,
    results: pd.DataFrame,
    metadata: Dict,
    centralized_objective=None,
    objective_tolerance=0.01,
    primal_tolerance=None,
    dual_tolerance=None,
    mismatch_tolerance=None,
    mode='combined',
    plot_rho=True,
    colors=None,
    ):
    """
    Plot the results and save them to `result_dir`.

    Args:
        - `result_dir`: Output directory.
        - `results`: `DataFrame` as returned by `input_output.read_results`.
        - `metadata`: Dict resembling an `AdmmMetadata` object.
        - `centralized_objective`: Centralized objective to compare against.
        - `objective_tolerance`: Desired objective gap.
        - `mode`: Save plots in `separate` plots, in a `combined` plot, or `both`.
        - `colors`: Dict of colors for the different plots. Supported keys are:
          primal, dual, mismatch, obj, rho.
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

    primal_tolerance = primal_tolerance or admmopt['primal_tolerance']
    dual_tolerance = dual_tolerance or admmopt['dual_tolerance']
    mismatch_tolerance = mismatch_tolerance or admmopt['mismatch_tolerance']

    n_clusters = len(metadata['clusters'])
    series = data_series(results, n_clusters, centralized_objective)
    series = series_cutoff(series, 10**(-4), 10**8)

    if combined:
        fig_combined, ax_combined = plt.subplots()
        ax_combined.set_yscale('log')
        ax_combined.set_xlabel('avg local iterations')
        ax_combined.set_title('Results per Iteration')

        primal_convergence = beginning_of_the_end(
            series['max_primal'], primal_tolerance
        )
        if primal_convergence >= 0:
            primal_convergence_iter = series['avg_iter'][primal_convergence]
            ax_combined.axvline(primal_convergence_iter, color=colors['primal'])

        dual_convergence = beginning_of_the_end(
            series['max_dual'], dual_tolerance
        )
        if dual_convergence >= 0:
            dual_convergence_iter = series['avg_iter'][dual_convergence]
            ax_combined.axvline(dual_convergence_iter, color=colors['dual'])

        mismatch_convergence = beginning_of_the_end(
            series['max_mismatch'], mismatch_tolerance
        )
        if mismatch_convergence >= 0:
            mismatch_convergence_iter = series['avg_iter'][mismatch_convergence]
            ax_combined.axvline(mismatch_convergence_iter, color=colors['mismatch'])

    if 'max_primal' in series:
        if separate:
            fig, ax = fig_primal()
            ax.axhline(primal_tolerance, color='black', linestyle='dashed')
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
            ax.axhline(dual_tolerance, color='black', linestyle='dashed')
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
            ax.axhline(mismatch_tolerance, color='black', linestyle='dashed')
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

            objective_convergence = beginning_of_the_end(series['obj_gap'], objective_tolerance)
            if objective_convergence >= 0:
                objective_convergence_iter = series['avg_iter'][objective_convergence]
                ax_combined.axvline(objective_convergence_iter, color=colors['obj'])

    if combined:
        if plot_rho:
            ax_combined.plot(series['avg_iter'], series['max_rho'], label='penalty', color=colors['rho'])
        ax_combined.legend()
        fig_combined.savefig(join(result_dir, 'combined.svg'))
        plt.close(fig_combined)

    solver_times = solver_times_dict(results, n_clusters)
    avg_times = { ID: sum(times) / len(times) for ID, times in solver_times.items() }
    avg_times = dict(sorted(avg_times.items(),key= lambda x:x[1], reverse=True))

    fig, ax = fig_avg_times()
    ax.bar(range(len(avg_times)), avg_times.values(), tick_label=list(avg_times.keys()))
    fig.savefig(join(result_dir, 'avg_times.svg'))
    plt.close(fig)


def plot_gaps(
    ax,
    color,
    label,
    results: pd.DataFrame,
    metadata: Dict,
    primal_tolerance: float,
    mismatch_tolerance: float,
    ):

    n_clusters = len(metadata['clusters'])
    series = data_series(results, n_clusters, None)
    series = series_cutoff(series, 10**(-4), 10**8)
    max_gap = [
        max(x, y) for x, y in zip(series['max_primal'], series['max_mismatch'])
    ]

    primal_convergence = beginning_of_the_end(
        series['max_primal'], primal_tolerance
    )
    mismatch_convergence = beginning_of_the_end(
        series['max_mismatch'], mismatch_tolerance
    )
    if primal_convergence >= 0 and mismatch_convergence >= 0:
        convergence = max(primal_convergence, mismatch_convergence)
        convergence_time = series['max_time'][convergence]
        ax.axvline(convergence_time, color=color)

    ax.plot(series['max_time'], max_gap, label=label, color=color)


def plot_objective(
    ax,
    color,
    label,
    results: pd.DataFrame,
    metadata: Dict,
    centralized_objective: float,
    objective_tolerance=0.01,
    ):

    n_clusters = len(metadata['clusters'])
    series = data_series(results, n_clusters, centralized_objective)
    series = series_cutoff(series, 10**(-4), 10**8)

    objective_convergence = beginning_of_the_end(series['obj_gap'], objective_tolerance)
    if objective_convergence >= 0:
        objective_convergence_iter = series['max_time'][objective_convergence]
        ax.axvline(objective_convergence_iter, color=color)

    ax.plot(series['max_time'], series['obj_gap'], label=label, color=color)


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


def fig_avg_times():
    fig, ax = plt.subplots()
    ax.set_xlabel('cluster')
    ax.set_ylabel('avg solver time (s)')
    ax.set_title('Avg solver time per cluster')
    return fig, ax
