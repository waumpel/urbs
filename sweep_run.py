﻿import argparse
from datetime import date
from multiprocessing import freeze_support, set_start_method
from os import makedirs, mkdir
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

    input_files = 'germany.xlsx'  # for single year file name, for intertemporal folder name
    input_dir = 'Input'
    input_path = join(input_dir, input_files)

    result_name = 'germany'
    if args.admm:
        result_name += '-admm-fixed'
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

    # simulation timesteps
    (offset, length) = (0, 1)  # time step selection
    timesteps = range(offset, offset + length + 1)
    dt = 1  # length of each time step (unit: hours)

    # select scenarios to be run
    scenarios = [
        scenario_base
    ]

    if args.admm:

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
            f'10^{i}': admm_async.AdmmOption(
                tolerance_mode = 'relative',
                primal_tolerance = 0.01,
                dual_tolerance = 0.01,
                mismatch_tolerance = 0.01,
                penalty_mode='fixed',
                rho = 10**i,
                max_iter = 200,
            )
            for i in range(4)
        }

        for scenario in scenarios:
            for opt_name, admmopt in admmopts.items():
                scenario_dir = join(result_dir, scenario.__name__, opt_name)
                makedirs(scenario_dir)

                year = date.today().year
                data_all = read_input(input_path, year)

                data_all, cross_scenario_data = scenario(data_all, cross_scenario_data)
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
                        cross_scenario_data=cross_scenario_data,
                    )
                else:
                    admm_objective = admm_async.run_parallel(
                    data_all,
                    timesteps,
                    scenario_dir,
                    dt,
                    objective,
                    clusters,
                    admmopt,
                    microgrid_cluster_mode='microgrid',
                    cross_scenario_data=cross_scenario_data,
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
            prob, cross_scenario_data = urbs.run_scenario(
                input_path,
                solver,
                timesteps,
                scenario,
                result_dir,
                dt,
                objective,
                cross_scenario_data=cross_scenario_data,
                threads=threads,
                report_tuples=report_tuples,
                report_sites_name=report_sites_name,
                plot_tuples=plot_tuples,
                plot_sites_name=plot_sites_name,
                plot_periods=plot_periods,
            )