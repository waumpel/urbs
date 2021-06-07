import argparse
from multiprocessing import freeze_support
import os
import shutil

from urbs import admm_async
from urbs.colorcodes import COLORS
from urbs.runfunctions import prepare_result_directory
from urbs.scenarios import scenario_base
from urbs.admm_async import plot
from urbs.admm_async import input_output

options = argparse.ArgumentParser()
options.add_argument('-c', '--centralized', action='store_true',
                     help='Additionally compute the centralized solution for comparison.')
args = options.parse_args()

# input_files = 'mimo-example_internal.xlsx'  # for single year file name, for intertemporal folder name
input_files = 'germany.xlsx'  # for single year file name, for intertemporal folder name
input_dir = 'Input'
input_path = os.path.join(input_dir, input_files)

result_name = 'Run'
result_dir = prepare_result_directory(result_name)  # name + time stamp

# copy input file to result directory
try:
    shutil.copytree(input_path, os.path.join(result_dir, input_dir))
except NotADirectoryError:
    shutil.copyfile(input_path, os.path.join(result_dir, input_files))
# copy run file to result directory
shutil.copy(__file__, result_dir)

# objective function
objective = 'cost'  # set either 'cost' or 'CO2' as objective

# simulation timesteps
(offset, length) = (0, 10)  # time step selection
timesteps = range(offset, offset + length + 1)
dt = 1  # length of each time step (unit: hours)

# clusters = [['Schleswig-Holstein','Hamburg','Mecklenburg-Vorpommern','Offshore','Lower Saxony','Bremen','Saxony-Anhalt','Brandenburg','Berlin','North Rhine-Westphalia'],
#                ['Baden-Württemberg','Hesse','Bavaria','Rhineland-Palatinate','Saarland','Saxony','Thuringia']]
clusters = [
    ['Schleswig-Holstein', 'Hamburg', 'Mecklenburg-Vorpommern', 'Offshore'],
    ['Lower Saxony', 'Bremen', 'Saxony-Anhalt', 'Brandenburg'],
    ['Berlin', 'North Rhine-Westphalia', 'Baden-Württemberg', 'Hesse'],
    ['Bavaria', 'Rhineland-Palatinate', 'Saarland', 'Saxony', 'Thuringia']
]
# clusters = [
#     ['Schleswig-Holstein'], ['Hamburg'], ['Mecklenburg-Vorpommern'], ['Offshore'],
#     ['Lower Saxony'], ['Bremen'], ['Saxony-Anhalt'], ['Brandenburg'],
#     ['Berlin'], ['North Rhine-Westphalia'], ['Baden-Württemberg'], ['Hesse'],
#     ['Bavaria'], ['Rhineland-Palatinate'], ['Saarland'], ['Saxony'], ['Thuringia']
# ]

# select scenarios to be run
scenarios = [
    scenario_base
]

admmopt = admm_async.AdmmOption(
    primal_tolerance = 0.01,
    dual_tolerance = 0.01,
    mismatch_tolerance = 0.01,
    rho = 1,
    max_penalty = 10**8,
    penalty_mult = 1.4,
    max_iter = 1000,
    tolerance_mode = 'relative',
    penalty_mode = 'increasing',
)

if __name__ == '__main__':
    freeze_support()
    for scenario in scenarios:
        data_all, ttime = admm_async.read(input_path, scenario, objective)

        admm_results = admm_async.run_regional(
            data_all = data_all,
            timesteps = timesteps,
            scenario = scenario,
            result_dir = result_dir,
            dt = dt,
            objective = objective,
            clusters = clusters,
            admmopt = admmopt,
        )

        if args.centralized:
            centralized_result = admm_async.run_centralized(
                data_all, timesteps, dt, scenario, result_dir
            )
            obj_cent = centralized_result['objective']
            obj_admm = admm_results['objective_values']['admm']
            gap = (obj_admm - obj_cent) / obj_cent
            admm_results['objective_values']['centralized'] = obj_cent
            admm_results['objective_values']['gap'] = gap
            admm_results['centralized_time'] = centralized_result['time']

        input_output.save_results(admm_results, result_dir)
        plot.plot_results(admm_results, result_dir, plot_rho=True)
