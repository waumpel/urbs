from multiprocessing import freeze_support
from os.path import join
import os
import shutil

from urbs import admm_async
from urbs.runfunctions import prepare_result_directory
from urbs.scenarios import scenario_base
from urbs.admm_async import plot
from urbs.admm_async import input_output

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
# clusters = [
#     ['Schleswig-Holstein', 'Hamburg', 'Mecklenburg-Vorpommern', 'Offshore'],
#     ['Lower Saxony', 'Bremen', 'Saxony-Anhalt', 'Brandenburg'],
#     ['Berlin', 'North Rhine-Westphalia', 'Baden-Württemberg', 'Hesse'],
#     ['Bavaria', 'Rhineland-Palatinate', 'Saarland', 'Saxony', 'Thuringia']
# ]
clusters = [
    ['Schleswig-Holstein'], ['Hamburg'], ['Mecklenburg-Vorpommern'], ['Offshore'],
    ['Lower Saxony'], ['Bremen'], ['Saxony-Anhalt'], ['Brandenburg'],
    ['Berlin'], ['North Rhine-Westphalia'], ['Baden-Württemberg'], ['Hesse'],
    ['Bavaria'], ['Rhineland-Palatinate'], ['Saarland'], ['Saxony'], ['Thuringia']
]

# select scenarios to be run
scenarios = [
    scenario_base
]

rhos = {i: 10**i for i in range(-1, 1)}
multiplier_values = [1 + 0.05 * i for i in range(1, 3)]
mults = {f'{v:.2f}': v for v in multiplier_values}
decreases_values = [0.5, 0.7, 0.9]
decs = {f'{v:.2f}': v for v in decreases_values}

print(f'sweeping over rhos: {list(rhos.values())}')
print(f'sweeping over mults: {list(mults.values())}')
print(f'sweeping over decs: {list(decs.values())}')

line = input('Continue? (Y/n): ')
if line not in ['', 'y']:
    print('Exiting')
    quit()

admmopts = {
    f'{rho_name}-{mult_name}-{dec_name}': admm_async.AdmmOption(
        primal_tolerance = 0.01,
        dual_tolerance = 0.01,
        mismatch_tolerance = 0.01,
        rho = rho,
        max_penalty = 10**8,
        penalty_mult = mult,
        primal_decrease = dec,
        max_iter = 100,
        tolerance_mode = 'relative',
    )
    for rho_name, rho in rhos.items()
    for mult_name, mult in mults.items()
    for dec_name, dec in decs.items()
}

if __name__ == '__main__':
    freeze_support()
    for scenario in scenarios:

        stats = []

        data_all, ttime = admm_async.read(input_path, scenario, objective)

        centralized_result = admm_async.run_centralized(
            data_all, timesteps, dt, scenario, result_dir
        )

        for name, admmopt in admmopts.items():
            print(f'Starting run {name}')
            sub_dir = os.path.join(result_dir, name)
            os.mkdir(sub_dir)

            admm_results = admm_async.run_regional(
                data_all = data_all,
                timesteps = timesteps,
                scenario = scenario.__name__,
                result_dir = sub_dir,
                dt = dt,
                objective = objective,
                clusters = clusters,
                admmopt = admmopt,
            )

            obj_cent = centralized_result['objective']
            obj_admm = admm_results['admm_objective']
            gap = (obj_admm - obj_cent) / obj_cent
            admm_results['centralized_objective'] = obj_cent
            admm_results['objective_gap'] = gap
            admm_results['centralized_time'] = centralized_result['time']

            stats.append({
                'name': name,
                'time': admm_results['admm_time'],
                'gap': 100 * abs(gap),
            })

            input_output.save_results(admm_results, sub_dir)
            plot.plot_results(admm_results, sub_dir, plot_rho=True)
            shutil.copy(join(sub_dir, 'combined.svg'), join(result_dir, f'{name}.svg'))

        input_output.save_overview(stats, result_dir)
