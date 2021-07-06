import json
from os.path import join


def cluster_results_dict(
        ID,
        regions,
        coupling_flows,
        timestamps,
        objective_values,
        primalgaps,
        dualgaps,
        max_mismatch_gaps,
        rhos,
        raw_dualgaps,
    ):
    return {
        'ID': ID,
        'regions': regions,
        'coupling_flows': coupling_flows,
        'iteration_series': [
            {
                'ID': ID,
                'time': timestamp,
                'obj': objective,
                'primal': primal,
                'dual': dual,
                'mismatch': mismatch,
                'rho': rho,
                'raw_dual': raw_dual,
            }
            for timestamp, objective, primal, dual, mismatch, rho, raw_dual in zip(
                timestamps,
                objective_values,
                primalgaps,
                dualgaps,
                max_mismatch_gaps,
                rhos,
                raw_dualgaps,
            )
        ]
    }


def results_dict(
    timesteps,
    scenario_name,
    dt,
    objective,
    clusters,
    admmopt,
    solver_time,
    objective_value,
    cluster_results,
):
    admmopt_dict = {
        attr: getattr(admmopt, attr)
        for attr in dir(admmopt) if not attr.startswith('__')
    }

    return {
        'timesteps': [timesteps.start, timesteps.stop],
        'scenario': scenario_name,
        'dt': dt,
        'objective': objective,
        'clusters': clusters,
        'admmopt': admmopt_dict,
        'admm_time': solver_time,
        'admm_objective': objective_value,
        'cluster_results': cluster_results,
    }


def results_path(result_dir):
    return join(result_dir, 'results.json')


def save_results(results_dict, result_dir,):
    path = results_path(result_dir)

    with open(path, 'w', encoding='utf8') as file:
        json.dump(results_dict, file, indent=4)


def load_results(result_dir):
    path = results_path(result_dir)

    with open(path, 'r', encoding='utf8') as file:
        results_dict = json.load(file)

    return results_dict


def overview_path(result_dir):
    return join(result_dir, 'overview.json')


def save_overview(stats, result_dir):
    """
    ## Arguments
    * stats: A list of dicts with the keys `name`, `time` and `gap`.
    """
    path = overview_path(result_dir)

    fastest = sorted(stats, key=lambda x: x['time'])[0]
    best = sorted(stats, key=lambda x: x['gap'])[0]
    good = filter(lambda x: x['gap'] < 1, stats)
    fastest_good = sorted(good, key=lambda x: x['time'])[0]

    overview = {
        'fastest': fastest,
        'best': best,
        'fastest_good': fastest_good,
    }

    with open(path, 'w', encoding='utf8') as file:
        json.dump(overview, file, indent=4)
