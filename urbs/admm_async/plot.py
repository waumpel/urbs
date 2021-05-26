from os.path import join

import matplotlib.pyplot as plt

def recent_results(results):
    """
    Return a list of the same length as `results`.
    The `i`th entry is a dict mapping each cluster to its most recent result in
    `results[:i+1]`.
    """
    first = results[0]
    recent_results = [{first['sender']: first}]
    for i in range(1, len(results)):
        d = recent_results[-1].copy()
        r = results[i]
        d[r['sender']] = r
        recent_results.append(d)


def plot_iteration_results(results, result_dir):
    times = [r['time'] for r in results]

    recent = recent_results(results)

    max_primal = [max(v['primalgap'] for v in d) for d in recent]
    max_dual = [max(v['dualgap'] for v in d) for d in recent]

    plot_primal_iter(max_primal, result_dir)
    plot_dual_iter(max_dual, result_dir)
    plot_primal_time(times, max_primal, result_dir)
    plot_dual_time(times, max_dual, result_dir)


def plot_primal_iter(max_primal, result_dir):
    fig, ax = plt.subplots()
    ax.plot(max_primal)
    ax.set_xlabel('sum of local iterations')
    ax.set_ylabel('max primal gap')
    ax.set_title('Primal Gap per Iteration')
    fig.savefig(join(result_dir, 'primal_iter.svg'))


def plot_dual_iter(max_dual, result_dir):
    fig, ax = plt.subplots()
    ax.plot(max_dual)
    ax.set_xlabel('sum of local iterations')
    ax.set_ylabel('max dual gap')
    ax.set_title('Dual Gap per Iteration')
    fig.savefig(join(result_dir, 'dual_iter.svg'))


def plot_primal_time(times, max_primal, result_dir):
    fig, ax = plt.subplots()
    ax.plot(times, max_primal)
    ax.set_xlabel('time')
    ax.set_ylabel('max primal gap')
    ax.set_title('Primal Gap over Time')
    fig.savefig(join(result_dir, 'primal_time.svg'))


def plot_dual_time(times, max_dual, result_dir):
    fig, ax = plt.subplots()
    ax.plot(times, max_dual)
    ax.set_xlabel('time')
    ax.set_ylabel('max dual gap')
    ax.set_title('Dual Gap over Time')
    fig.savefig(join(result_dir, 'dual_time.svg'))
