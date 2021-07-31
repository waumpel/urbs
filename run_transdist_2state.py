# TODO: fix this once run_transdist_1state is working

# -*- coding: utf-8 -*-
import argparse
from datetime import date
from multiprocessing import freeze_support
import os
from os.path import join
import shutil
from urbs.validation import validate_dc_objective, validate_input
from urbs.input import read_input
import urbs
import pandas as pd

from urbs import admm_async
from urbs.runfunctions import prepare_result_directory
from urbs.admm_async import plot
from urbs.admm_async import input_output

options = argparse.ArgumentParser()
options.add_argument('-c', '--centralized', action='store_true',
                     help='Additionally compute the centralized solution for comparison.')
args = options.parse_args()

input_files = 'transdist-2state.xlsx'  # for single year file name, for intertemporal folder name
# input_files = 'Transmission_Level.xlsx'  # for single year file name, for intertemporal folder name
input_dir = 'Input'
input_path = os.path.join(input_dir, input_files)

microgrid_files = ['Microgrid_rural_A.xlsx', 'Microgrid_urban_A.xlsx']
microgrid_dir = 'Input/Microgrid_types'
microgrid_paths = [
    os.path.join(microgrid_dir, file)
    for file in microgrid_files
]
result_name = 'transdist-2state'
result_dir = prepare_result_directory(result_name)  # name + time stamp

# #copy input file to result directory
try:
    shutil.copytree(input_path, os.path.join(result_dir, input_dir))
except NotADirectoryError:
    shutil.copyfile(input_path, os.path.join(result_dir, input_files))

# #copy run file to result directory
shutil.copy(__file__, result_dir)

# objective function
objective = 'cost'  # set either 'cost' or 'CO2' as objective

# simulation timesteps
(offset, length) = (0, 1)  # time step selection
timesteps = range(offset, offset+length+1)
dt = 1  # length of each time step (unit: hours)

clusters = [['BB', 'MV']]

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

if __name__ == '__main__':
    freeze_support()
    for scenario in scenarios:
        year = date.today().year
        data_all = read_input(input_path, year)

        with open(join(result_dir, 'timevar-main.txt'), 'w', encoding='utf8') as f:
            f.write(data_all['eff_factor'].to_string())

        data_all, cross_scenario_data = scenario(data_all, cross_scenario_data)
        validate_input(data_all)
        validate_dc_objective(data_all, objective)

        admm_results = admm_async.run_regional(
            data_all = data_all,
            timesteps = timesteps,
            scenario = scenario.__name__,
            result_dir = result_dir,
            dt = dt,
            objective = objective,
            clusters = clusters,
            admmopt = admmopt,
            microgrid_files = microgrid_paths,
            cross_scenario_data = cross_scenario_data,
            microgrid_cluster_mode = 'microgrid',
        )

        if args.centralized:
            # run_regional already performed transdist preprocessing,
            # don't have to do that again (TODO: factor out preprocessing)
            centralized_result = admm_async.run_centralized(
                data_all, timesteps, dt, scenario, result_dir
            )
            obj_cent = centralized_result['objective']
            obj_admm = admm_results['admm_objective']
            gap = (obj_admm - obj_cent) / obj_cent
            admm_results['centralized_objective'] = obj_cent
            admm_results['objective_gap'] = gap
            admm_results['centralized_time'] = centralized_result['time']

        input_output.save_results(admm_results, result_dir)
        plot.plot_results(admm_results, result_dir, plot_rho=True)

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
