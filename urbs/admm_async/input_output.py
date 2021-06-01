import json
from os.path import join


def attr_dict(obj):
    return {
        attr: getattr(obj, attr)
        for attr in dir(obj) if not attr.startswith('__')
    }


def make_results_dict(
        input_file,
        timesteps,
        scenario,
        dt,
        objective,
        clusters,
        admmopt,
        times,
        objective_values,
        results,
    ):
    return {
        'input file' : input_file,
        'timesteps': [timesteps.start, timesteps.stop],
        'scenario': scenario.__name__,
        'dt': dt,
        'objective': objective,
        'clusters': clusters,
        'admmopt': attr_dict(admmopt),
        'times': times,
        'objective_values': objective_values,
        'results': results,
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
