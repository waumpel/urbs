from os.path import join

import matplotlib.pyplot as plt


# TODO: actual merge sort implementation
def merge_sort_results(results):
    """
    Merge the results of individual clusters and sort them by timestamp.
    Return a list of dicts.

    `results` is a list of dicts, each representing all results from one cluster.
    """
    return sorted((
        {
            'ID': r['ID'],
            'time': timestamp,
            'obj': obj,
            'primal': primal,
            'dual': dual,
            'mismatch': mismatch,
        }
        for r in results
        for timestamp, obj, primal, dual, mismatch in zip(
            r['timestamps'],
            r['objective'],
            r['primal_residual'],
            r['dual_residual'],
            r['constraint_mismatch'],
        )),
        key=lambda x: x['time']
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


def plot_results(results_dict, result_dir):
    """
    Plot the results and save them to `result_dir`.

    `results_dict` is a dict as returned by `.input_output.make_results_dict`.
    """
    n_clusters = len(results_dict['clusters'])
    admmopt = results_dict['admmopt']
    results = merge_sort_results(results_dict['results'])
    recent = recent_results(results)

    max_primal = [max(v['primal'] for v in d.values()) for d in recent]
    max_dual = [max(v['dual'] for v in d.values()) for d in recent]
    max_mismatch = [max(v['mismatch'] for v in d.values()) for d in recent]
    avg_iter = [x / n_clusters for x in range(len(max_primal))]

    fig, ax = fig_primal()
    ax.axhline(admmopt['primal_tolerance'], color='black', linestyle='dashed')
    ax.plot(avg_iter, max_primal)
    fig.savefig(join(result_dir, 'primal.svg'))

    fig, ax = fig_dual()
    ax.axhline(admmopt['dual_tolerance'], color='black', linestyle='dashed')
    ax.plot(avg_iter, max_dual)
    fig.savefig(join(result_dir, 'dual.svg'))

    fig, ax = fig_mismatch()
    ax.axhline(admmopt['mismatch_tolerance'], color='black', linestyle='dashed')
    ax.plot(avg_iter, max_mismatch)
    fig.savefig(join(result_dir, 'mismatch.svg'))

    objective_values = results_dict['objective_values']
    if 'centralized' in objective_values:
        sum_obj = [sum(v['obj'] for v in d.values()) for d in recent]
        centralized_obj = objective_values['centralized'] if 'centralized' in objective_values else None
        obj_gap = [abs(x - centralized_obj) / centralized_obj for x in sum_obj]

        fig, ax = fig_objective()
        ax.axhline(0.01, color='black', linestyle='dashed')
        ax.plot(avg_iter, obj_gap)
        fig.savefig(join(result_dir, 'objective.svg'))


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
