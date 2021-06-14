from os.path import join

import matplotlib.pyplot as plt


# TODO: actual merge sort implementation
def merge_sort_results(cluster_results):
    """
    Merge the iteration_results of individual clusters and sort them by timestamp.
    Return a list of dicts. Each dict holds the results of one iteration from one cluster.
    """
    return sorted(
        (
            iteration_result
            for cluster_results_dict in cluster_results
            for iteration_result in cluster_results_dict['iteration_series']
        ),
        key=lambda d: d['time']
    )


def recent_results(results):
    """
    Return a list of the same length as `results`.
    The `i`th entry is a dict mapping each cluster to its most recent result in
    `results[:i+1]`.

    `results` is a list of dicts, each representing a single iteration result.
    """
    first = results[0]
    recent_results = [{first['ID']: first}]
    for i in range(1, len(results)):
        d = recent_results[-1].copy()
        r = results[i]
        d[r['ID']] = r
        recent_results.append(d)

    return recent_results


def data_series(results_dict):
    n_clusters = len(results_dict['clusters'])
    results = merge_sort_results(results_dict['cluster_results'])
    recent = recent_results(results)

    series = {}
    first = list(recent[0].values())[0] # dict of the first cluster

    if 'primal' in first:
        max_primal = [max(v['primal'] for v in d.values()) for d in recent]
        series['max_primal'] = max_primal
    if 'dual' in first:
        max_dual = [max(v['dual'] for v in d.values()) for d in recent]
        series['max_dual'] = max_dual
    if 'mismatch' in first:
        max_mismatch = [max(v['mismatch'] for v in d.values()) for d in recent]
        series['max_mismatch'] = max_mismatch
    if 'obj' in first:
        sum_obj = [sum(v['obj'] for v in d.values()) for d in recent]
        series['sum_obj'] = sum_obj
    if 'rho' in first:
        max_rho = [max(v['rho'] for v in d.values()) for d in recent]
        series['max_rho'] = max_rho
    if 'raw_dual' in first:
        max_raw_dual = [max(v['raw_dual'] for v in d.values()) for d in recent]
        series['max_raw_dual'] = max_raw_dual

    avg_iter = [x / n_clusters for x in range(len(max_primal))]
    series['avg_iter'] = avg_iter

    return series


def series_cutoff(series, cutoff):
    new_series = {}
    for name, data in series.items():
        if name == 'avg_iter':
            new_series[name] = data
        new_series[name] = [max(cutoff, x) for x in data]
    return new_series


def plot_results(results_dict, result_dir, plot_rho=False):
    """
    Plot the results and save them to `result_dir`.

    `results_dict` is a dict as returned by `.runfunctions_admm.run_regional`.
    """
    admmopt = results_dict['admmopt']
    series = series_cutoff(data_series(results_dict), 10**(-4))

    fig_combined, ax_combined = plt.subplots()
    ax_combined.set_yscale('log')
    ax_combined.set_xlabel('avg local iterations')
    ax_combined.set_title('Results per Iteration')

    if 'max_primal' in series:
        fig, ax = fig_primal()
        ax.axhline(admmopt['primal_tolerance'], color='black', linestyle='dashed')
        ax.plot(series['avg_iter'], series['max_primal'])
        ax_combined.plot(series['avg_iter'], series['max_primal'], label='primal gap')
        if plot_rho:
            ax.plot(series['avg_iter'], series['max_rho'], color='black')
        fig.savefig(join(result_dir, 'primal.svg'))
        plt.close(fig)

    if 'max_dual' in series:
        fig, ax = fig_dual()
        ax.axhline(admmopt['dual_tolerance'], color='black', linestyle='dashed')
        ax.plot(series['avg_iter'], series['max_dual'])
        ax_combined.plot(series['avg_iter'], series['max_dual'], label='dualgap')
        if plot_rho:
            ax.plot(series['avg_iter'], series['max_rho'], color='black')
        fig.savefig(join(result_dir, 'dual.svg'))
        plt.close(fig)

    if 'max_mismatch' in series:
        fig, ax = fig_mismatch()
        ax.axhline(admmopt['mismatch_tolerance'], color='black', linestyle='dashed')
        ax.plot(series['avg_iter'], series['max_mismatch'])
        ax_combined.plot(series['avg_iter'], series['max_mismatch'], label='mismatch')
        if plot_rho:
            ax.plot(series['avg_iter'], series['max_rho'], color='black')
        fig.savefig(join(result_dir, 'mismatch.svg'))
        plt.close(fig)

    if 'max_raw_dual' in series:
        fig, ax = fig_raw_dual()
        ax.plot(series['avg_iter'], series['max_raw_dual'])
        if plot_rho:
            ax.plot(series['avg_iter'], series['max_rho'], color='black')
        fig.savefig(join(result_dir, 'raw_dual.svg'))
        plt.close(fig)

    if 'sum_obj' in series:
        if 'centralized_objective' in results_dict:
            centralized_obj = results_dict['centralized_objective']
            obj_gap = [abs(x - centralized_obj) / centralized_obj for x in series['sum_obj']]

            fig, ax = fig_objective()
            ax.axhline(0.01, color='black', linestyle='dashed')
            ax.plot(series['avg_iter'], obj_gap)
            ax_combined.plot(series['avg_iter'], obj_gap, label='objective gap')
            if plot_rho:
                ax.plot(series['avg_iter'], series['max_rho'], color='black')
            fig.savefig(join(result_dir, 'objective.svg'))
            plt.close(fig)

    if plot_rho:
        ax_combined.plot(series['avg_iter'], series['max_rho'], label='penalty')

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
