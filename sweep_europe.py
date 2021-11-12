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
    options.add_argument('-s', '--sequential', action='store_true')
    args = options.parse_args()

    input_files = 'europe-intertemporal'  # for single year file name, for intertemporal folder name
    input_dir = 'Input'
    input_path = join(input_dir, input_files)

    # simulation timesteps
    (offset, length) = (0, 1)  # time step selection
    timesteps = range(offset, offset + length + 1)
    dt = 1  # length of each time step (unit: hours)

    result_name = f'europe-intertemporal-t{length}'
    if args.sequential:
        result_name += '-seq'
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

    # select scenarios to be run
    scenarios = [
        scenario_base
    ]

    # ADMM clusters
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
        f'rho={rho}-mult={mult}-dec={dec}-alpha={alpha}': admm_async.AdmmOption(
            penalty_mode='increasing',
            rho=rho,
            max_penalty=10**8,
            penalty_mult=mult,
            primal_decrease=dec,
            max_iter=500,
            tolerance=(0.01, None, 0.01),
            async_correction=alpha,
        )
        for rho in reversed([10**i for i in range(-1, 4)])
        for mult in [1.5]
        for dec in [0.95]
        for alpha in [0]
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

            if args.sequential:
                admm_async.run_sequential(
                    'gurobi',
                    data_all,
                    timesteps,
                    opt_dir,
                    dt,
                    objective,
                    clusters,
                    admmopt,
                    microgrid_cluster_mode='microgrid',
                )
            else:
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
