from datetime import date
from multiprocessing import freeze_support, set_start_method
from os import mkdir
from os.path import join, isdir
import shutil
import time

import urbs
from urbs import admm_async
from urbs.input import read_input
from urbs.runfunctions import prepare_result_directory
from urbs.validation import validate_dc_objective, validate_input

if __name__ == '__main__':
    set_start_method("spawn")
    freeze_support()

    input_files = 'Transmission_Level.xlsx'  # for single year file name, for intertemporal folder name
    input_dir = 'Input'
    input_path = join(input_dir, input_files)

    microgrid_files = ['Microgrid_rural_A.xlsx', 'Microgrid_urban_A.xlsx']
    microgrid_dir = 'Input/Microgrid_types'
    microgrid_paths = [
        join(microgrid_dir, file)
        for file in microgrid_files
    ]
    result_name = 'transdist-tsam-admm'
    result_dir = prepare_result_directory(result_name)  # name + time stamp

    # #copy input file to result directory
    try:
        shutil.copytree(input_path, join(result_dir, input_dir))
    except NotADirectoryError:
        shutil.copyfile(input_path, join(result_dir, input_files))

    # #copy run file to result directory
    shutil.copy(__file__, result_dir)

    # objective function
    objective = 'cost'  # set either 'cost' or 'CO2' as objective

    # simulation timesteps
    (offset, length) = (0, 1)  # time step selection
    timesteps = range(offset, offset+length+1)
    dt = 1  # length of each time step (unit: hours)

    clusters = [
        ['BB', 'BE', 'BW', 'BY', 'HB', 'HE', 'HH', 'MV', 'NI', 'NW', 'RP', 'SH', 'SL', 'SN', 'ST', 'TH', 'Baltic', 'North']
    ]

    # input data for tsam method
    noTypicalPeriods = 4
    hoursPerPeriod = 168

    threads = 1

    # select scenarios to be run
    scenarios = [
                urbs.transdist100, # transdist100 scenarios must be simulated first to store distribution demand
                #urbs.transdist66,
                #urbs.transdist33,
                #urbs.transdist75,
                #urbs.transdist50,
                #urbs.transdist25,
                #urbs.transmission
                ]

    admmopt = admm_async.AdmmOption(
        primal_tolerance = 0.01,
        dual_tolerance = 0.01,
        mismatch_tolerance = 0.01,
        rho = 1,
        max_penalty = 10**8,
        penalty_mult = 1.1,
        primal_decrease = 0.9,
        # residual_distance = 10,
        # mult_adapt = 1,
        # max_mult = 10**8,
        max_iter = 200,
        tolerance_mode = 'relative',
    )

    cross_scenario_data = {}

    for scenario in scenarios:
        print(f'Running scenario {scenario.__name__}')
        scenario_start = time.time()
        scenario_dir = join(result_dir, scenario.__name__)
        if not isdir(scenario_dir):
            mkdir(scenario_dir)

        year = date.today().year
        data_all = read_input(input_path, year)

        data_all, cross_scenario_data = scenario(data_all, cross_scenario_data)
        validate_input(data_all)
        validate_dc_objective(data_all, objective)

        admm_objective = admm_async.run_regional(
            data_all,
            timesteps,
            scenario_dir,
            dt,
            objective,
            clusters,
            admmopt,
            microgrid_files=microgrid_paths,
            microgrid_cluster_mode='microgrid',
            cross_scenario_data=cross_scenario_data,
            noTypicalPeriods=noTypicalPeriods,
            hoursPerPeriod=hoursPerPeriod,
            threads=threads,
        )

        print(f'Time elapsed for scenario {scenario.__name__}: {(time.time() - scenario_start):.2f} s')

            # TODO: how to get `prob`, used to be model instance (in centralized approach)
            # if scenario.__name__ == 'transdist100':
            #     cap_PV_private = prob._result['cap_pro'].loc[:, :, 'PV_private_rooftop'].droplevel(level=[0])
            #     cap_PV_private.index = pd.MultiIndex.from_tuples(cap_PV_private.index.str.split('_').tolist())
            #     cap_PV_private = cap_PV_private.groupby(level=[2]).sum().to_frame()
            #     cap_PV_private.index.name = 'sit'
            #     cap_PV_private['pro'] = 'PV_private_rooftop'
            #     cap_PV_private.set_index(['pro'], inplace=True, append=True)
            #     cap_PV_private = cap_PV_private.squeeze()
            #     cross_scenario_data['PV_cap_shift'] = cap_PV_private
