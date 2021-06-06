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
    times = [r['time'] for r in results]
    recent = recent_results(results)

    max_primal = [max(v['primal'] for v in d.values()) for d in recent]
    max_dual = [max(v['dual'] for v in d.values()) for d in recent]
    max_mismatch = [max(v['mismatch'] for v in d.values()) for d in recent]
    sum_obj = [sum(v['obj'] for v in d.values()) for d in recent]
    avg_iter = [x / n_clusters for x in range(len(max_primal))]

    plot_primal_iter(max_primal, avg_iter, admmopt['primal_tolerance'], result_dir)
    plot_dual_iter(max_dual, avg_iter, admmopt['dual_tolerance'], result_dir)
    plot_mismatch_iter(max_mismatch, avg_iter, admmopt['mismatch_tolerance'], result_dir)
    plot_primal_time(times, max_primal, admmopt['primal_tolerance'], result_dir)
    plot_dual_time(times, max_dual, admmopt['dual_tolerance'], result_dir)

    objective_values = results_dict['objective_values']
    centralized_obj = objective_values['centralized'] if 'centralized' in objective_values else None
    plot_obj_iter(sum_obj, avg_iter, centralized_obj, result_dir)

    if 'centralized' in objective_values:
        obj_gap = [abs(x - centralized_obj) / centralized_obj for x in sum_obj]
        plot_obj_rel_iter(obj_gap, avg_iter, result_dir)


def plot_primal_iter(max_primal, avg_iter, primal_tolerance, result_dir):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(avg_iter, max_primal)
    ax.axhline(primal_tolerance, color='black', linestyle='dashed')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('max primal gap')
    ax.set_title('Primal Gap per Iteration')
    fig.savefig(join(result_dir, 'primal_iter.svg'))


def plot_dual_iter(max_dual, avg_iter, dual_tolerance, result_dir):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(avg_iter, max_dual)
    ax.axhline(dual_tolerance, color='black', linestyle='dashed')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('max dual gap')
    ax.set_title('Dual Gap per Iteration')
    fig.savefig(join(result_dir, 'dual_iter.svg'))


def plot_mismatch_iter(max_mismatch, avg_iter, mismatch_tolerance, result_dir):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(avg_iter, max_mismatch)
    ax.axhline(mismatch_tolerance, color='black', linestyle='dashed')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('max mismatch gap')
    ax.set_title('Mismatch Gap per Iteration')
    fig.savefig(join(result_dir, 'mismatch_iter.svg'))



def plot_primal_time(times, max_primal, primal_tolerance, result_dir):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(times, max_primal)
    ax.axhline(primal_tolerance, color='black', linestyle='dashed')
    ax.set_xlabel('time')
    ax.set_ylabel('max primal gap')
    ax.set_title('Primal Gap over Time')
    fig.savefig(join(result_dir, 'primal_time.svg'))


def plot_dual_time(times, max_dual, dual_tolerance, result_dir):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(times, max_dual)
    ax.axhline(dual_tolerance, color='black', linestyle='dashed')
    ax.set_xlabel('time')
    ax.set_ylabel('max dual gap')
    ax.set_title('Dual Gap over Time')
    fig.savefig(join(result_dir, 'dual_time.svg'))


def plot_obj_iter(sum_obj, avg_iter, centralized_obj, result_dir):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(avg_iter, sum_obj)
    if centralized_obj:
        ax.axhline(centralized_obj, color='black', linestyle='dashed')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('objective value')
    ax.set_title('Objective Value per Iteration')
    fig.savefig(join(result_dir, 'obj_iter.svg'))


def plot_obj_rel_iter(obj_gap, avg_iter, result_dir):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(avg_iter, obj_gap)
    ax.axhline(0.01, color='black', linestyle='dashed')
    ax.set_xlabel('avg local iterations')
    ax.set_ylabel('relative objective gap')
    ax.set_title('Relative Objective Gap per Iteration')
    fig.savefig(join(result_dir, 'obj_rel_iter.svg'))
