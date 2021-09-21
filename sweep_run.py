import argparse
from datetime import date
from multiprocessing import freeze_support, set_start_method
from os import makedirs
from os.path import join
import shutil

from urbs import admm_async
from urbs.input import read_input
from urbs.runfunctions import prepare_result_directory
from urbs.scenarios import scenario_base
from urbs.validation import validate_dc_objective, validate_input


if __name__ == '__main__':
    set_start_method("spawn")
    freeze_support()

    options = argparse.ArgumentParser()
    options.add_argument('-s', '--sequential', action='store_true')
    args = options.parse_args()

    input_files = 'germany.xlsx'  # for single year file name, for intertemporal folder name
    input_dir = 'Input'
    input_path = join(input_dir, input_files)

    result_name = 'germany-t1-admm'
    if args.sequential:
        result_name += '-seq'
    result_name += '-inc'
    result_dir = prepare_result_directory(result_name)  # name + time stamp

    # copy input file to result directory
    try:
        shutil.copytree(input_path, join(result_dir, input_dir))
    except NotADirectoryError:
        shutil.copyfile(input_path, join(result_dir, input_files))

    # copy run file to result directory
    shutil.copy(__file__, result_dir)

    # objective function
    objective = 'cost'  # set either 'cost' or 'CO2' as objective

    # simulation timesteps
    (offset, length) = (0, 1)  # time step selection
    timesteps = range(offset, offset + length + 1)
    dt = 1  # length of each time step (unit: hours)

    # select scenarios to be run
    scenarios = [
        scenario_base
    ]

    # one cluster
    # clusters = [['Schleswig-Holstein','Hamburg','Mecklenburg-Vorpommern','Offshore','Lower Saxony','Bremen','Saxony-Anhalt','Brandenburg','Berlin','North Rhine-Westphalia'],
    #                ['Baden-Württemberg','Hesse','Bavaria','Rhineland-Palatinate','Saarland','Saxony','Thuringia']]

    # four clusters
    # clusters = [
    #     ['Schleswig-Holstein', 'Hamburg', 'Mecklenburg-Vorpommern', 'Offshore'],
    #     ['Lower Saxony', 'Bremen', 'Saxony-Anhalt', 'Brandenburg'],
    #     ['Berlin', 'North Rhine-Westphalia', 'Baden-Württemberg', 'Hesse'],
    #     ['Bavaria', 'Rhineland-Palatinate', 'Saarland', 'Saxony', 'Thuringia']
    # ]

    # one cluster per state
    clusters = [
        ['Schleswig-Holstein'], ['Hamburg'], ['Mecklenburg-Vorpommern'], ['Offshore'],
        ['Lower Saxony'], ['Bremen'], ['Saxony-Anhalt'], ['Brandenburg'],
        ['Berlin'], ['North Rhine-Westphalia'], ['Baden-Württemberg'], ['Hesse'],
        ['Bavaria'], ['Rhineland-Palatinate'], ['Saarland'], ['Saxony'], ['Thuringia']
    ]

    admmopts = {
        f'rho={rho}-mult={mult}-dec={dec}': admm_async.AdmmOption(
            tolerance_mode = 'relative',
            primal_tolerance = 0.01,
            dual_tolerance = 0.01,
            mismatch_tolerance = 0.01,
            penalty_mode='increasing',
            rho = rho,
            max_penalty=10**8,
            penalty_mult=mult,
            primal_decrease=dec,
            max_iter = 200,
        )
        for rho in [100]
        for mult in [1.15]
        for dec in [0.95]
    }

    for scenario in scenarios:
        results = {}
        for opt_name, admmopt in admmopts.items():
            scenario_dir = join(result_dir, scenario.__name__, opt_name)
            makedirs(scenario_dir)

            year = date.today().year
            data_all = read_input(input_path, year)

            data_all, _ = scenario(data_all)
            validate_input(data_all)
            validate_dc_objective(data_all, objective)

            if args.sequential:
                admm_async.run_sequential(
                    'gurobi',
                    data_all,
                    timesteps,
                    scenario_dir,
                    dt,
                    objective,
                    clusters,
                    admmopt,
                    microgrid_cluster_mode='microgrid',
                )
            else:
                result = admm_objective = admm_async.run_parallel(
                data_all,
                timesteps,
                scenario_dir,
                dt,
                objective,
                clusters,
                admmopt,
                microgrid_cluster_mode='microgrid',
                )
                results[opt_name] = result

        summaries = []
        for opt_name, result in results.items():
            if result['converged']:
                summaries.append(f"{opt_name}: {result['avg iterations']} {result['time']} {result['objective']}")
            else:
                summaries.append(f'{opt_name}: - - -')

        print(f'Results for scenario {scenario.__name__}:')
        with open(join(scenario_dir, 'summary.txt'), 'w', encoding='utf8') as f:
            for s in summaries:
                print(s)
                f.write(s + '\n')
