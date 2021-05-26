import json
from os.path import join

def save_iteration_results(results, result_dir):
    path = join(result_dir, 'iteration_results.json')

    results.sort(key=lambda x: x['time'])
    t_offset = results[0]['time']
    for result in results:
        result['time'] -= t_offset

    with open(path, 'w', encoding='utf8') as file:
        json.dump(results, file, indent=4)
