from datetime import date, datetime
import json
import multiprocessing as mp
import os
from os.path import join
import time
from urbs.admm_async.admm_worker import AdmmWorker
from urbs.admm_async.admm_messages import AdmmStatus, AdmmStatusMessage, AdmmIterationResult
from urbs.features.typeperiod import run_tsam
from urbs.features.transdisthelper import *
from urbs.identify import identify_mode

import numpy as np
import pandas as pd
import psutil as ps
from pyomo.environ import SolverFactory

import urbs.model
from urbs.input import read_input, add_carbon_supplier
from urbs.validation import validate_input
from.admm_metadata import AdmmMetadata


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


def run_regional(
    data_all,
    timesteps,
    result_dir,
    dt,
    objective,
    clusters,
    admmopt,
    microgrid_files=None,
    microgrid_cluster_mode='microgrid',
    cross_scenario_data=None,
    noTypicalPeriods=None,
    hoursPerPeriod=None,
    threads=1,
    ):
    """
    Run an urbs model for given input, time steps and scenario with regional decomposition
    using ADMM.

    Args:
        - `data_all`: Input data dict, after applying scenario and validation.
        - `timesteps`: List of timesteps, e.g. range(0,8761).
        - `result_dir`: Directory name for result spreadsheet and plots.
        - `dt`: Length of each time step in hours.
        - `objective`: Objective function, either "cost" or "CO2".
        - `clusters`: List of lists partitioning the sites of the problem into clusters.
        - `admmopt`: `AdmmOption` object.
        - `microgrid_files`: Filenames of input Excel spreadsheets for microgrid types.
        - `microgrid_cluster_mode`: If `microgrid`, one cluster per microgrid is added.
          If `all`, one cluster for ALL microgrid nodes is added. Default: `microgrid`.
        - `cross_scenario_data`: Dict for storing data across scenarios
        - `noTypicalPeriods`: Number of typical periods (TSAM parameter)
        - `hoursPerPeriod`: Length of each typical period (TSAM parameter)
        - `threads`: Number of Gurobi threads to use PER CLUSTER. Default: 1.

    Return the value of the objective function.
    """
    # hard-coded year. ADMM doesn't work with intertemporal models (yet)
    year = date.today().year

    # read and modify microgrid data
    mode = identify_mode(data_all)
    if mode['transdist']:
        microgrid_data_initial =[]
        for i, microgrid_file in enumerate(microgrid_files):
            microgrid_data_initial.append(read_input(microgrid_file, year))
            validate_input(microgrid_data_initial[i])
        # join microgrid data to model data
        microgrid_nodes = create_transdist_data(data_all, microgrid_data_initial, cross_scenario_data)

        if microgrid_cluster_mode == 'all':
            # put ALL microgrid nodes into one cluster
            microgrid_cluster = [
                node
                for microgrid in microgrid_nodes
                for node in microgrid
            ]
            clusters.append(microgrid_cluster)
            print('Added one microgrid cluster for all microgrid nodes.')
        elif microgrid_cluster_mode == 'microgrid':
            # create one cluster for every microgrid
            clusters.extend(microgrid_nodes.to_list())
            print('Added one cluster per microgrid.')
        else:
            raise ValueError(f"Unsupported `microgrid_cluster_mode`; must be 'all' or 'microgrid'.")

    elif mode['acpf']:
        add_reactive_transmission_lines(data_all)
        add_reactive_output_ratios(data_all)

    if mode['tsam']:
        timesteps, weighting_order = run_tsam(
            data_all, noTypicalPeriods, hoursPerPeriod, cross_scenario_data)
    else:
        weighting_order = None

    print('ADMM preprocessing')

    # add carbon supplier if necessary
    if not np.isinf(data_all['global_prop'].loc[year].loc['CO2 limit', 'value']):
        add_carbon_supplier(data_all, clusters)
        clusters.append(['Carbon_site'])
        print("Added carbon supplier cluster.")

    # store metadata
    metadata = AdmmMetadata(clusters, admmopt)
    with open(join(result_dir, 'metadata.json'), 'w', encoding='utf8') as f:
        json.dump(metadata.to_dict(), f, indent=4)

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

    # Queue for collecting the results from each subproblem after convergence
    output = manager.Queue()

    procs = []
    for ID in range(n_clusters):
        proc = mp.Process(
            name=f'AdmmWorker[{ID}]',
            target=AdmmWorker.run_worker,
            args=(
                ID,
                result_dir,
                output,
                data_all,
                timesteps,
                dt,
                objective,
                year,
                initial_values,
                admmopt,
                n_clusters,
                clusters[ID],
                neighbors[ID],
                shared_lines[ID],
                internal_lines[ID],
                cluster_from[ID],
                cluster_to[ID],
                neighbor_cluster[ID],
                queues,
                hoursPerPeriod,
                weighting_order,
                threads,
            )
        )
        procs.append(proc)

    print('Spawning worker processes')

    solver_start = time.time()
    for proc in procs:
        proc.start()

    model_creation_status = { ID: False for ID in range(n_clusters) }
    while not all(model_creation_status.values()):
        msg = output.get(block=True)
        if msg['msg'] == 'model created':
            model_creation_status[msg['sender']] = True
        else:
            RuntimeWarning(f'Received unexpected msg')

    print('All workers have created their models. Starting ADMM')

    memory = ps.Process().memory_info().rss + sum(
        ps.Process(proc.pid).memory_info().rss for proc in procs
    )
    print(f'Currently using {(memory / 10**9):.2f} GiB of memory (rss)')

    for q in queues:
        q.put('start solving')

    status = [ None ] * n_clusters
    results = [ None ] * n_clusters

    with open(join(result_dir, 'iteration_results.txt'), 'w', encoding='utf8') as logfile:
        logfile.write(' '.join(AdmmIterationResult._HEADERS) + '\n')
        while True:
            msg = output.get(block=True)
            if isinstance(msg, AdmmIterationResult):
                msg.subtract_time(solver_start)
                logfile.write(str(msg) + '\n')
                logfile.flush()
                results[msg.process_id] = msg
                print_status(results, status)
            elif isinstance(msg, AdmmStatusMessage):
                status[msg.sender] = msg.status
                if all(s in [AdmmStatus.TERMINATED, AdmmStatus.GLOBAL_CONVERGENCE]
                       for s in status):
                    print_status(results, status)
                    break
            else:
                RuntimeWarning(f'Received item of unexpected type, ' +
                                'should be `AdmmIterationResult` or `AdmmStatusMsg`')

    for proc in procs:
        proc.join()

    ttime = time.time()
    solver_time = ttime - solver_start

    admm_objective = sum(result.objective for result in results)

    # print results
    print(f'ADMM solver time: {solver_time:4.0f} s')
    print(f'ADMM objective  : {admm_objective:.4e}')

    return admm_objective


def print_status(results, status):
    status_map = {
        AdmmStatus.TERMINATED: 'terminated',
        AdmmStatus.GLOBAL_CONVERGENCE: 'converged',
        None: 'running',
    }
    df = pd.DataFrame(
        columns=['iteration', 'time', 'status'],
        data=[
            [
                result.local_iteration if result else '-',
                result.stop_time - result.start_time if result else '-',
                status_map[value]
            ]
            for result, value in zip(results, status)
        ]
    )
    print('\n' + df.to_string())
