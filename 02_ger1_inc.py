import argparse
from datetime import date
from multiprocessing import freeze_support, set_start_method
from os import makedirs
from os.path import join, realpath, dirname
import shutil

from urbs import admm_async
from urbs.input import read_input
from urbs.runfunctions import prepare_result_directory
from urbs.scenarios import scenario_base


if __name__ == '__main__':
    set_start_method("spawn")
    freeze_support()

    options = argparse.ArgumentParser()
    args = options.parse_args()

    input_files = 'germany.xlsx'  # for single year file name, for intertemporal folder name
    input_dir = 'Input'
    input_path = join(input_dir, input_files)

    result_name = '02_ger1_inc'
    result_dir = prepare_result_directory(result_name)  # name + time stamp

    # copy input file to result directory
    try:
        shutil.copytree(input_path, join(result_dir, input_dir))
    except NotADirectoryError:
        shutil.copyfile(input_path, join(result_dir, input_files))

    # copy run file to result directory
    shutil.copy(__file__, result_dir)

    # simulation timesteps
    (offset, length) = (0, 1)  # time step selection
    timesteps = range(offset, offset + length + 1)
    dt = 1  # length of each time step (unit: hours)

    # objective function
    objective = 'cost'  # set either 'cost' or 'CO2' as objective

    # select scenarios to be run
    scenarios = [
        scenario_base
    ]

    # one cluster per state
    clusters = [
        ['Schleswig-Holstein'], ['Hamburg'], ['Mecklenburg-Vorpommern'], ['Offshore'],
        ['Lower Saxony'], ['Bremen'], ['Saxony-Anhalt'], ['Brandenburg'],
        ['Berlin'], ['North Rhine-Westphalia'], ['Baden-Württemberg'], ['Hesse'],
        ['Bavaria'], ['Rhineland-Palatinate'], ['Saarland'], ['Saxony'], ['Thuringia']
    ]

    admmopts = {
        f'rho={rho}-tau={tau}-xi={xi}': admm_async.AdmmOption(
            penalty_mode='increasing',
            rho=rho,
            penalty_mult=tau,
            primal_decrease=xi,
            async_correction=0,
            max_penalty=10**8,
            max_iter=200,
            tolerance=0.0,
        )
        for rho in [10**i for i in range(1, 6)]
        for tau in [1.1, 1.5, 2, 5, 10]
        for xi in [0.5, 0.9, 0.95, 0.99]
    }

    year = date.today().year
    data_all = read_input(input_path, year)

    for scenario in scenarios:
        print(f'\nStarting scenario {scenario.__name__}')
        data_all, _ = scenario(data_all)

        scenario_dir = join(result_dir, scenario.__name__)
        makedirs(scenario_dir)
        shutil.copy(join(dirname(realpath(__file__)), 'plot.py'), scenario_dir)
        results = {}

        for opt_name, admmopt in admmopts.items():
            print(f'\nStarting admmopt {opt_name}')
            opt_dir = join(scenario_dir, opt_name)
            makedirs(opt_dir)

            result = admm_async.run_parallel(
                data_all,
                timesteps,
                opt_dir,
                dt,
                objective,
                clusters,
                admmopt,
                microgrid_cluster_mode='microgrid',
            )
            results[opt_name] = result
