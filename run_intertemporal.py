import argparse
from datetime import date
from multiprocessing import freeze_support, set_start_method
from os import mkdir
from os.path import join, isdir
import shutil

import urbs
from urbs import admm_async
from urbs.input import read_input
from urbs.runfunctions import prepare_result_directory
from urbs.scenarios import scenario_base
from urbs.validation import validate_dc_objective, validate_input


if __name__ == '__main__':
    set_start_method("spawn")
    freeze_support()

    options = argparse.ArgumentParser()
    options.add_argument('-a', '--admm', action='store_true')
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
    if args.admm:
        result_name += '-admm'
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

    if args.admm:
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

        admmopt = admm_async.AdmmOption(
            rho = 10000,
            scale_rho = True,
            max_penalty = 10**8,
            penalty_mult = 2,
            primal_decrease = 0.95,
            # residual_distance = 10,
            # mult_adapt = 1,
            # max_mult = 10**8,
            max_iter = 500,
            tolerance = (0.01, None, 0.01),
        )

        for scenario in scenarios:
            scenario_dir = join(result_dir, scenario.__name__)
            if not isdir(scenario_dir):
                mkdir(scenario_dir)

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
                result = admm_async.run_parallel(
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
        solver = 'gurobi'
        threads = 1

        report_tuples = []
        report_sites_name = {}
        plot_tuples = []
        plot_sites_name = {}
        plot_periods = {
            'all': timesteps[1:]
        }

        for scenario in scenarios:
            prob, _ = urbs.run_scenario(
                input_path,
                solver,
                timesteps,
                scenario,
                result_dir,
                dt,
                objective,
                threads=threads,
                report_tuples=report_tuples,
                report_sites_name=report_sites_name,
                plot_tuples=plot_tuples,
                plot_sites_name=plot_sites_name,
                plot_periods=plot_periods,
            )
