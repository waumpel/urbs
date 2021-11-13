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

    input_files = 'europe'  # for single year file name, for intertemporal folder name
    input_dir = 'Input'
    input_path = join(input_dir, input_files)

    result_name = '08_eur1000'
    result_dir = prepare_result_directory(result_name)  # name + time stamp

    # copy input file to result directory
    try:
        shutil.copytree(input_path, join(result_dir, input_dir))
    except NotADirectoryError:
        shutil.copyfile(input_path, join(result_dir, input_files))

    # copy run file to result directory
    shutil.copy(__file__, result_dir)

    # simulation timesteps
    (offset, length) = (0, 1000)  # time step selection
    timesteps = range(offset, offset + length + 1)
    dt = 1  # length of each time step (unit: hours)

    # objective function
    objective = 'cost'  # set either 'cost' or 'CO2' as objective

    # select scenarios to be run
    scenarios = [
        scenario_base
    ]

    clusters = [
        ['ALB'],
        ['AUT'],
        ['BEL'],
        ['BGR'],
        ['BIH'],
        ['CHE'],
        ['CZE'],
        ['DEU'],
        ['DNK'],
        ['ESP'],
        ['EST'],
        ['FIN'],
        ['FRA'],
        ['GBR'],
        ['GRC'],
        ['HRV'],
        ['HUN'],
        ['IRL'],
        ['ITA'],
        ['KOS'],
        ['LTU'],
        ['LUX'],
        ['LVA'],
        ['MKD'],
        ['MNE'],
        ['NLD'],
        ['NOR'],
        ['POL'],
        ['PRT'],
        ['ROU'],
        ['SRB'],
        ['SVK'],
        ['SVN'],
        ['SWE'],
    ]

    admmopts = {
        f'rho={rho}': admm_async.AdmmOption(
            penalty_mode='increasing',
            rho=rho,
            penalty_mult=1.5,
            primal_decrease=0.95,
            async_correction=0,
            max_penalty=10**8,
            max_iter=500,
            tolerance=(0.01, None, 0.01),
        )
        for rho in [10**i for i in range(-2, 4)]
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
