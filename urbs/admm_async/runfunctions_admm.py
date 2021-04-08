from datetime import date, datetime
from math import ceil
import multiprocessing as mp
import os
import queue
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.environ import SolverFactory

from urbs.model import create_model
from urbs.input import read_input, add_carbon_supplier
from urbs.validation import validate_dc_objective, validate_input
from .run_worker import run_worker
from .urbs_admm_model import UrbsAdmmModel


class CouplingVars:

    def __init__(self):
        self.flow_global = {}
        self.rhos = {}
        self.lambdas = {}
        self.cap_global = {}
        self.residdual = {}
        self.residprim = {}


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


def setup_solver(optim, logfile='solver.log'):
    """ """
    if optim.name == 'gurobi':
        # reference with list of option names
        # http://www.gurobi.com/documentation/5.6/reference-manual/parameters
        optim.set_options("logfile={}".format(logfile))
        optim.set_options("method=2")
        # optim.set_options("timelimit=7200")  # seconds
        # optim.set_options("mipgap=5e-4")  # default = 1e-4
    elif optim.name == 'glpk':
        # reference with list of options
        # execute 'glpsol --help'
        optim.set_options("log={}".format(logfile))
        # optim.set_options("tmlim=7200")  # seconds
        # optim.set_options("mipgap=.0005")
    elif optim.name == 'cplex':
        optim.set_options("log={}".format(logfile))
    else:
        print("Warning from setup_solver: no options set for solver "
              "'{}'!".format(optim.name))
    return optim


# @profile
def run_regional(input_file, timesteps, scenario, result_dir,
                 dt, objective, clusters=None):
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
    # hard-coded year. ADMM doesn't work with intertemporal models (yet)
    year = date.today().year

    # scenario name, read and modify data for scenario
    sce = scenario.__name__
    data_all = read_input(input_file, year)
    data_all = scenario(data_all)
    validate_input(data_all)
    validate_dc_objective(data_all, objective)

    if not data_all['global_prop'].loc[year].loc['CO2 limit', 'value'] == np.inf:
        data_all = add_carbon_supplier(data_all, clusters)
        clusters.append(['Carbon_site'])

    # if 'test_timesteps' is stored in data dict, replace the timesteps parameter with that value
    timesteps = data_all.pop('test_timesteps', timesteps)

    # initiate a coupling-variables Class
    coup_vars = CouplingVars()

    # identify the shared and internal lines

    nclusters = len(clusters)

    # map site -> cluster_idx
    site_cluster_map = {}
    for cluster, cluster_idx in zip(clusters, range(nclusters)):
        for site in cluster:
            site_cluster_map[site] = cluster_idx

    # used as indices for creating `shared_lines` and `internal_lines`
    shared_lines_logic = np.zeros((nclusters, data_all['transmission'].shape[0]), dtype=bool)
    internal_lines_logic = np.zeros((nclusters, data_all['transmission'].shape[0]), dtype=bool)

    # Source/target cluster of each shared line for each cluster.
    # These are appended as additional columns to `shared_lines` along with `neighbor_cluster` (defined below).
    cluster_from = [[] for _ in range(nclusters)]
    cluster_to = [[] for _ in range(nclusters)]

    # Set of neighbors for each cluster
    neighbors = [set() for _ in range(nclusters)]

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

    # map cluster_idx -> slice of data_all['transmission']
    shared_lines = [
        data_all['transmission'].loc[shared_lines_logic[cluster_idx, :]]
        for cluster_idx in range(0, nclusters)
    ]
    # map cluster_idx -> slice of data_all['transmission']
    internal_lines = [
        data_all['transmission'].loc[internal_lines_logic[cluster_idx, :]]
        for cluster_idx in range(0, nclusters)
    ]
    # neighbouring cluster of each shared line for each cluster
    neighbor_cluster = [
        np.array(cluster_from[cluster_idx]) + np.array(cluster_to[cluster_idx]) - cluster_idx
        for cluster_idx in range(0, nclusters)
    ]

    # initialize coupling variables
    for cluster_idx in range(0, nclusters):
        for i in range(0, shared_lines[cluster_idx].shape[0]):
            sit_from = shared_lines[cluster_idx].iloc[i].name[1]
            sit_to = shared_lines[cluster_idx].iloc[i].name[2]

            for j in timesteps[1:]:
                coup_vars.lambdas[cluster_idx, j, year, sit_from, sit_to] = 0
                coup_vars.rhos[cluster_idx, j, year, sit_from, sit_to] = 5
                coup_vars.flow_global[cluster_idx, j, year, sit_from, sit_to] = 0


    # (optional) create the central problem to compare results
    prob = create_model(data_all, timesteps, dt, type='normal')

    # refresh time stamp string and create filename for logfile
    log_filename = os.path.join(result_dir, '{}.log').format(sce)

    # setup solver
    solver_name = 'gurobi'
    optim = SolverFactory(solver_name)  # cplex, glpk, gurobi, ...
    optim = setup_solver(optim, logfile=log_filename)

    # original problem solution (not necessary for ADMM, to compare results)
    orig_time_before_solve = time.time()
    results_prob = optim.solve(prob, tee=False)
    orig_time_after_solve = time.time()
    orig_duration = orig_time_after_solve - orig_time_before_solve
    flows_from_original_problem = dict((name, entity.value) for (name, entity) in prob.e_tra_in.items())
    flows_from_original_problem = pd.DataFrame.from_dict(flows_from_original_problem, orient='index',
                                                         columns=['Original'])

    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999

    problems = []
    sub = {}

    # initiate urbs_admm_model Classes for each subproblem
    for cluster_idx in range(0, nclusters):
        problem = UrbsAdmmModel()
        sub[cluster_idx] = create_model(data_all, timesteps, type='sub',
                                             sites=clusters[cluster_idx],
                                             coup_vars=coup_vars,
                                             data_transmission_boun=shared_lines[cluster_idx],
                                             data_transmission_int=internal_lines[cluster_idx],
                                             cluster=cluster_idx)
        problem.sub_pyomo = sub[cluster_idx]
        problem.flow_global = {(key[1], key[2], key[3], key[4]): value
                               for (key, value) in coup_vars.flow_global.items() if key[0] == cluster_idx}
        problem.flow_global = pd.Series(problem.flow_global)
        problem.flow_global.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)
        problem.flow_global = problem.flow_global.to_frame()

        problem.lamda = {(key[1], key[2], key[3], key[4]): value
                         for (key, value) in coup_vars.lambdas.items() if key[0] == cluster_idx}
        problem.lamda = pd.Series(problem.lamda)
        problem.lamda.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)
        problem.lamda = problem.lamda.to_frame()

        problem.rho = 5

        problem.ID = cluster_idx
        problem.result_dir = result_dir
        problem.sce = sce
        # enlarge shared_lines (copies of slices of data_all['transmission'])
        shared_lines[cluster_idx]['cluster_from'] = cluster_from[cluster_idx]
        shared_lines[cluster_idx]['cluster_to'] = cluster_to[cluster_idx]
        shared_lines[cluster_idx]['neighbor_cluster'] = neighbor_cluster[cluster_idx]
        problem.shared_lines = shared_lines[cluster_idx]
        problem.na = nclusters
        problems.append(problem)

    # create Queues for each communication channel
    queues = {
        source: {
            target: mp.Manager().Queue() # TODO: is creation of multiple managers intended?
            for target in neighbors[source]
        }
        for source in range(nclusters)
    }

    # define further necessary fields for the subproblems
    for cluster_idx in range(0, nclusters):
        problem = problems[cluster_idx]
        problem.neighbors = neighbors[cluster_idx]

        problem.nneighbors = len(problem.neighbors)

        problem.sending_queues = queues[cluster_idx]
        problem.receiving_queues = {
            target: queues[target][cluster_idx]
            for target in neighbors[cluster_idx]
        }

        problem.nwait = ceil(
            problem.nneighbors * problem.admmopt.nwaitPercent)

    # define a Queue class for collecting the results from each subproblem after convergence
    output = mp.Manager().Queue()

    # define the asynchronous jobs for ADMM routines
    procs = []
    for cluster_idx in range(0, nclusters):
        procs += [mp.Process(target=run_worker, args=(cluster_idx + 1, problems[cluster_idx], output))]

    start_time = time.time()
    start_clock = time.clock()
    for proc in procs:
        proc.start()

    liveprocs = list(procs)

    # collect results as the subproblems converge
    results = []
    while liveprocs:
        try:
            while 1:
                results.append(output.get(False))
        except queue.Empty:
            pass

        time.sleep(0.5)
        if not output.empty():
            continue

        liveprocs = [p for p in liveprocs if p.is_alive()]

    for proc in procs:
        proc.join()

    # ------------get results ---------------------------
    ttime = time.time()
    tclock = time.clock()
    totaltime = ttime - start_time
    clocktime = tclock - start_clock

    results = sorted(results, key=lambda x: x[0])

    obj_total = 0
    obj_cent = results_prob['Problem'][0]['Lower bound']

    for cluster_idx in range(0, nclusters):
        if cluster_idx != results[cluster_idx][0]:
            print('Error: Result of worker %d not returned!' % (cluster_idx + 1,))
            break
        obj_total += results[cluster_idx][1]['cost']

        # debug
        received_neighbors = results[cluster_idx][1]['received_neighbors']
        print('cluster', cluster_idx, 'received neighbors:', received_neighbors, 'avg:', sum(received_neighbors)/len(received_neighbors))

    gap = (obj_total - obj_cent) / obj_cent * 100
    print('The convergence time for original problem is %f' % (orig_duration,))
    print('The convergence time for ADMM is %f' % (totaltime,))
    print('The convergence clock time is %f' % (clocktime,))
    print('The objective function value is %f' % (obj_total,))
    print('The central objective function value is %f' % (obj_cent,))
    print('The gap in objective function is %f %%' % (gap,))

    #testlog
    file_object = open('log_for_test.txt', 'a')
    file_object.write('Timesteps for this test is %f' % (len(timesteps),))
    file_object.write('The convergence time for original problem is %f' % (orig_duration,))
    file_object.write('The convergence time for ADMM is %f' % (totaltime,))
    file_object.write('The gap in objective function is %f %%' % (gap,))
    file_object.close()
    # ------------ plots of convergence -----------------
    fig = plt.figure()
    for cluster_idx in range(0, nclusters):
        if cluster_idx != results[cluster_idx][0]:
            print('Error: Result of worker %d not returned!' % (cluster_idx + 1,))
            break
        pgap = results[cluster_idx][1]['primal_residual']
        dgap = results[cluster_idx][1]['dual_residual']
        curfig = fig.add_subplot(1, nclusters, cluster_idx + 1)
        curfig.plot(pgap, color='red', linewidth=2.5, label='primal residual')
        curfig.plot(dgap, color='blue', linewidth=2.5, label='dual residual')
        curfig.set_yscale('log')
        curfig.legend(loc='upper right')

    #plt.show()

    return sub