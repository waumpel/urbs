from datetime import date, datetime
import multiprocessing as mp
import os
from os.path import join
from time import time, clock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.environ import SolverFactory

from urbs.model import create_model
from urbs.input import read_input, add_carbon_supplier
from urbs.validation import validate_dc_objective, validate_input
from .input_output import save_iteration_results
from .plot import plot_iteration_results
from .run_worker import run_worker
from .urbs_admm_model import UrbsAdmmModel


class InitialValues:
    """
    Holds the initial values for several variables and parameters.
    Intended use: Each member holds a scalar value that is used for all values in a
    `pd.Series` or `pd.DataFrame`.

    ### Members:
    * `flow`
    * `flow_global`
    * `lamda`
    """

    def __init__(self, flow, flow_global, lamda):
        self.flow = flow
        self.flow_global = flow_global
        self.lamda = lamda


def prepare_result_directory(result_name):
    """ create a time stamped directory within the result folder.

    Args:
        result_name: user specified result name

    Returns:
        a subfolder in the result folder

    """
    # timestamp for result directory
    now = datetime.now().strftime('%Y%m%dT%H%M')

    # create result directory if not existent
    result_dir = os.path.join('result', '{}-{}'.format(result_name, now))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir


def setup_solver(solver, logfile='solver.log'):
    """ """
    if solver.name == 'gurobi':
        # reference with list of option names
        # http://www.gurobi.com/documentation/5.6/reference-manual/parameters
        solver.set_options("logfile={}".format(logfile))
        solver.set_options("method=2")
        # solver.set_options("timelimit=7200")  # seconds
        # solver.set_options("mipgap=5e-4")  # default = 1e-4
    elif solver.name == 'glpk':
        # reference with list of options
        # execute 'glpsol --help'
        solver.set_options("log={}".format(logfile))
        # solver.set_options("tmlim=7200")  # seconds
        # solver.set_options("mipgap=.0005")
    elif solver.name == 'cplex':
        solver.set_options("log={}".format(logfile))
    else:
        print("Warning from setup_solver: no options set for solver "
              "'{}'!".format(solver.name))
    return solver


# @profile
def run_regional(
    input_file,
    timesteps,
    scenario,
    result_dir,
    dt,
    objective,
    clusters,
    admmopt,
    centralized=False
    ):
    """ run an urbs model for given input, time steps and scenario with regional decomposition using ADMM

    Args:
        input_file: filename to an Excel spreadsheet for urbs.read_excel
        timesteps: a list of timesteps, e.g. range(0,8761)
        scenario: a scenario function that modifies the input data dict
        result_dir: directory name for result spreadsheet and plots
        dt: width of a time step in hours(default: 1)
        objective: the entity which is optimized ('cost' of 'co2')
        clusters: user-defined region clusters for regional decomposition (list of lists)

    Returns:
        the urbs model instances
    """

    logfile = open(join(result_dir, 'shared.log'), 'w', encoding='utf8')
    def log(*args):
        msg = ' '.join(str(arg) for arg in args)
        print(msg)
        logfile.write(msg + '\n')

    # hard-coded year. ADMM doesn't work with intertemporal models (yet)
    year = date.today().year

    # scenario name, read and modify data for scenario
    scenario_name = scenario.__name__

    log('Reading input...')
    start = time()
    data_all = read_input(input_file, year)
    read_time = time() - start
    log(f'Done. Time elapsed: {read_time:.2f} seconds')

    log('Preprocessing...')
    start = time()

    data_all = scenario(data_all)
    validate_input(data_all)
    validate_dc_objective(data_all, objective)

    if not data_all['global_prop'].loc[year].loc['CO2 limit', 'value'] == np.inf:
        data_all = add_carbon_supplier(data_all, clusters)
        clusters.append(['Carbon_site'])

    # if 'test_timesteps' is stored in data dict, replace the timesteps parameter with that value
    timesteps = data_all.pop('test_timesteps', timesteps)

    n_clusters = len(clusters)

    # map site -> cluster_idx
    site_cluster_map = {}
    for cluster, cluster_idx in zip(clusters, range(n_clusters)):
        for site in cluster:
            site_cluster_map[site] = cluster_idx

    # Note:
    # data_all['transmission'].index:
    # ['support_timeframe', 'Site In', 'Site Out', 'Transmission', 'Commodity']

    # identify the shared and internal lines

    # used as indices for creating `shared_lines` and `internal_lines`
    shared_lines_logic = np.zeros((n_clusters, data_all['transmission'].shape[0]), dtype=bool)
    internal_lines_logic = np.zeros((n_clusters, data_all['transmission'].shape[0]), dtype=bool)

    # Source/target cluster of each shared line for each cluster.
    # These are appended as additional columns to `shared_lines` along with `neighbor_cluster` (defined below).
    cluster_from = [[] for _ in range(n_clusters)]
    cluster_to = [[] for _ in range(n_clusters)]

    # Set of neighbors for each cluster
    neighbors = [set() for _ in range(n_clusters)]

    for row, (_, site_in, site_out, tra, com) in zip(range(0, data_all['transmission'].shape[0]), data_all['transmission'].index):
        from_cluster_idx = site_cluster_map[site_in]
        to_cluster_idx = site_cluster_map[site_out]

        if from_cluster_idx != to_cluster_idx:
            # shared line
            neighbors[from_cluster_idx].add(to_cluster_idx)
            neighbors[to_cluster_idx].add(from_cluster_idx)
            shared_lines_logic[from_cluster_idx, row] = True
            shared_lines_logic[to_cluster_idx, row] = True
            cluster_from[from_cluster_idx].append(from_cluster_idx)
            cluster_to[from_cluster_idx].append(to_cluster_idx)
            cluster_from[to_cluster_idx].append(from_cluster_idx)
            cluster_to[to_cluster_idx].append(to_cluster_idx)
        else:
            # internal line
            internal_lines_logic[from_cluster_idx, row] = True
            internal_lines_logic[to_cluster_idx, row] = True

    # map cluster_idx -> slice of data_all['transmission'] (copies)
    shared_lines = [
        data_all['transmission'].loc[shared_lines_logic[cluster_idx, :]].copy(deep=True)
        for cluster_idx in range(0, n_clusters)
    ]
    # map cluster_idx -> slice of data_all['transmission'] (copies)
    internal_lines = [
        data_all['transmission'].loc[internal_lines_logic[cluster_idx, :]].copy(deep=True)
        for cluster_idx in range(0, n_clusters)
    ]
    # neighbouring cluster of each shared line for each cluster
    neighbor_cluster = [
        np.array(cluster_from[cluster_idx]) + np.array(cluster_to[cluster_idx]) - cluster_idx
        for cluster_idx in range(0, n_clusters)
    ]

    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999

    initial_values = InitialValues(
        flow=0,
        flow_global=0,
        lamda=0,
    )

    # Manager object for creating Queues
    manager = mp.Manager()

    # Queues for communication between processes
    queues = [manager.Queue() for _ in range(n_clusters)]

    problems = []
    sub = {}

    preprocess_time = time() - start
    log(f'Done. Time elapsed: {preprocess_time:.0f}')

    log('Creating models...')
    start = time()

    # initialize pyomo models and `UrbsAdmmModel`s
    for cluster_idx in range(0, n_clusters):
        index = shared_lines[cluster_idx].index.to_frame()

        flow_global = pd.Series({
            (t, year, source, target): initial_values.flow_global
            for t in timesteps[1:]
            for source, target in zip(index['Site In'], index['Site Out'])
        })
        flow_global.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)

        lamda = pd.Series({
            (t, year, source, target): initial_values.lamda
            for t in timesteps[1:]
            for source, target in zip(index['Site In'], index['Site Out'])
        })
        lamda.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)

        model = create_model(data_all, timesteps, type='sub',
                             sites=clusters[cluster_idx],
                             data_transmission_boun=shared_lines[cluster_idx],
                             data_transmission_int=internal_lines[cluster_idx],
                             flow_global=flow_global,
                             lamda=lamda,
                             rho=admmopt.rho)

        sub[cluster_idx] = model

        sending_queues = {
            target: queues[target] for target in neighbors[cluster_idx]
        }

        # enlarge shared_lines (copies of slices of data_all['transmission'])
        shared_lines[cluster_idx]['cluster_from'] = cluster_from[cluster_idx]
        shared_lines[cluster_idx]['cluster_to'] = cluster_to[cluster_idx]
        shared_lines[cluster_idx]['neighbor_cluster'] = neighbor_cluster[cluster_idx]

        problem = UrbsAdmmModel(
            admmopt = admmopt,
            flow_global = flow_global,
            ID = cluster_idx,
            lamda = lamda,
            model = model,
            n_clusters = n_clusters,
            neighbors = neighbors[cluster_idx],
            receiving_queue = queues[cluster_idx],
            regions = clusters[cluster_idx],
            result_dir = result_dir,
            scenario_name = scenario_name,
            sending_queues = sending_queues,
            shared_lines = shared_lines[cluster_idx],
            shared_lines_index = index,
        )

        problems.append(problem)

    model_time = time() - start
    log(f'Done. Time elapsed: {model_time:.0f}')

    # Queue for collecting the results from each subproblem after convergence
    output = manager.Queue()

    # Queue for accumulating log messages
    logqueue = manager.Queue()

    iteration_results_queue = manager.Queue()

    # Child processes for the ADMM subproblems
    procs = [
        mp.Process(target=run_worker, args=(problem, output, logqueue, iteration_results_queue))
        for problem in problems
    ]

    log('Solving the distributed problem...')

    start_time = time()
    start_clock = clock()
    for proc in procs:
        proc.start()

    # collect results as the subproblems converge
    results = [
        output.get(block=True) for _ in range(n_clusters)
    ]

    for proc in procs:
        proc.join()

    ttime = time()
    tclock = clock()
    solver_time = ttime - start_time
    solver_clock = tclock - start_clock

    # Write accumulated log messages to logfile
    while not logqueue.empty():
        msg = logqueue.get(block=False)
        logfile.write(msg + '\n')

    log(f'Done. Time elapsed: {solver_time:.0f}')

    # get results
    results = sorted(results, key=lambda x: x[0])

    obj_total = 0

    for cluster_idx in range(0, n_clusters):
        if cluster_idx != results[cluster_idx][0]:
            raise RuntimeError(f'Result of worker {cluster_idx + 1} was not returned')
        obj_total += results[cluster_idx][1]['cost'][-1]

    # (optinal) solve the centralized problem
    if centralized:
        log('Creating the centralized model...')
        start = time()
        prob = create_model(data_all, timesteps, dt, type='normal')
        centralized_model_time = time() - start
        log(f'Done. Time elapsed: {centralized_model_time:.0f}')

        # refresh time stamp string and create filename for logfile
        log_filename = os.path.join(result_dir, f'{scenario_name}.log')

        # setup solver
        solver_name = 'gurobi'
        solver = SolverFactory(solver_name)  # cplex, glpk, gurobi, ...
        solver = setup_solver(solver, logfile=log_filename)

        log('Solving the centralized model...')
        start = time()
        result = solver.solve(prob, tee=False)
        centralized_solver_time = time() - start
        log(f'Done. Time elapsed: {centralized_solver_time:.0f}')
        flows_from_original_problem = pd.DataFrame.from_dict(
            {name: entity.value for name, entity in prob.e_tra_in.items()},
            orient='index',
            columns=['Original']
        )

        obj_cent = result['Problem'][0]['Lower bound']

    # print results
    log()
    log(f'Reading input time      : {read_time:4.0f} s')
    log(f'ADMM preprocessing time : {preprocess_time:4.0f} s')
    log(f'ADMM model creation time: {model_time:4.0f} s')
    log(f'ADMM solver time        : {solver_time:4.0f} s')
    log(f'ADMM solver clock       : {solver_clock:4.0f} s')
    log(f'ADMM objective          : {obj_total:.4e}')

    if centralized:
        gap = (obj_total - obj_cent) / obj_cent
        log()
        log(f'centralized model creation time: {centralized_model_time:4.0f} s')
        log(f'centralized solver time        : {centralized_solver_time:4.0f} s')
        log(f'centralized objective          : {obj_cent:.4e}')
        log()
        log(f'Objective gap: {gap:.4%}')

    logfile.close()

    iteration_results = []
    while not iteration_results_queue.empty():
        iteration_results.append(iteration_results_queue.get())

    save_options(admmopt)
    save_iteration_results(iteration_results, result_dir)
    plot_iteration_results(iteration_results, result_dir)

    # ------------ plots of convergence -----------------
    # fig = plt.figure()
    # for cluster_idx in range(0, n_clusters):
    #     if cluster_idx != results[cluster_idx][0]:
    #         print('Error: Result of worker %d not returned!' % (cluster_idx + 1,))
    #         break
    #     pgap = results[cluster_idx][1]['primal_residual']
    #     dgap = results[cluster_idx][1]['dual_residual']
    #     curfig = fig.add_subplot(1, n_clusters, cluster_idx + 1)
    #     curfig.plot(pgap, color='red', linewidth=2.5, label='primal residual')
    #     curfig.plot(dgap, color='blue', linewidth=2.5, label='dual residual')
    #     curfig.set_yscale('log')
    #     curfig.legend(loc='upper right')

    #plt.show()

    return sub
