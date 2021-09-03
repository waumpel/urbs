from datetime import date, datetime
import json
import multiprocessing as mp
import os
from os.path import join
import pickle
from time import time
from urbs.model import cost_rule_sub
from urbs.runfunctions import setup_solver

import numpy as np
import pandas as pd
import psutil as ps
import pyomo.environ as pyomo

import urbs
from urbs.admm_async.admm_model import AdmmModel
from urbs.admm_async.admm_worker import AdmmWorker
from urbs.admm_async.admm_messages import AdmmStatus, AdmmStatusMessage, AdmmIterationResult, AdmmVariableMessage
from urbs.features.typeperiod import run_tsam
from urbs.features.transdisthelper import *
from urbs.identify import identify_mode
from urbs.input import read_input, add_carbon_supplier
from urbs.validation import validate_input
from.admm_metadata import AdmmMetadata


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


def prepare_admm(
    data_all,
    timesteps,
    clusters,
    microgrid_files=None,
    microgrid_cluster_mode='microgrid',
    cross_scenario_data=None,
    noTypicalPeriods=None,
    hoursPerPeriod=None,
    ):
    """
    Run an urbs model for given input, time steps and scenario with regional decomposition
    using ADMM.

    Args:
        - `data_all`: Input data dict, after applying scenario and validation.
        - `timesteps`: List of timesteps, e.g. range(0,8761).
        - `clusters`: List of lists partitioning the sites of the problem into clusters.
        - `microgrid_files`: Filenames of input Excel spreadsheets for microgrid types.
        - `microgrid_cluster_mode`: If `microgrid`, one cluster per microgrid is added.
          If `all`, one cluster for ALL microgrid nodes is added. Default: `microgrid`.
        - `cross_scenario_data`: Dict for storing data across scenarios
        - `noTypicalPeriods`: Number of typical periods (TSAM parameter)
        - `hoursPerPeriod`: Length of each typical period (TSAM parameter)

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
        timesteps, weighting_order = run_tsam(data_all, noTypicalPeriods, hoursPerPeriod,
                                              cross_scenario_data)
    else:
        weighting_order = None

    print('ADMM preprocessing')

    # add carbon supplier if necessary
    if not np.isinf(data_all['global_prop'].loc[year].loc['CO2 limit', 'value']):
        add_carbon_supplier(data_all, clusters)
        clusters.append(['Carbon_site'])
        print("Added carbon supplier cluster.")

    n_clusters = len(clusters)

    # map site -> cluster ID
    site_cluster_map = {}
    for cluster, ID in zip(clusters, range(n_clusters)):
        for site in cluster:
            site_cluster_map[site] = ID

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

    for row, (_, site_in, site_out, tra, com) in enumerate(data_all['transmission'].index):
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

    # map ID -> slice of data_all['transmission'] (copies)
    internal_lines = [
        data_all['transmission'].loc[internal_lines_logic[ID, :]].copy(deep=True)
        for ID in range(n_clusters)
    ]

    # neighbouring cluster of each shared line for each cluster
    neighbor_cluster = [
        np.array(cluster_from[ID]) + np.array(cluster_to[ID]) - ID
        for ID in range(n_clusters)
    ]

    # map ID -> slice of data_all['transmission'] (copies)
    shared_lines = [
        data_all['transmission'].loc[shared_lines_logic[ID, :]].copy(deep=True)
        for ID in range(n_clusters)
    ]
    # enlarge shared_lines (copies of slices of data_all['transmission'])
    for ID in range(n_clusters):
        shared_lines[ID]['cluster_from'] = cluster_from[ID]
        shared_lines[ID]['cluster_to'] = cluster_to[ID]
        shared_lines[ID]['neighbor_cluster'] = neighbor_cluster[ID]

    shared_lines_index = [
        shared_lines[ID].index.to_frame()
        for ID in range(n_clusters)
    ]

    initial_flow_global = 0
    initial_lamda = 0

    flow_global = [
        fill_flow_global(year, timesteps, shared_lines_index[ID], initial_flow_global)
        for ID in range(n_clusters)
    ]

    lamda = [
        fill_lamda(year, timesteps, shared_lines_index[ID], initial_lamda)
        for ID in range(n_clusters)
    ]

    return (
        timesteps,
        clusters,
        neighbors,
        internal_lines,
        shared_lines,
        shared_lines_index,
        flow_global,
        lamda,
        weighting_order,
    )


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


def fill_flow_global(year, timesteps, shared_lines_index, value):
    flow_global = pd.Series({
        (t, year, source, target): value
        for t in timesteps[1:]
        for source, target in zip(
            shared_lines_index['Site In'], shared_lines_index['Site Out']
        )
    })
    flow_global.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)
    return flow_global


def fill_lamda(year, timesteps, shared_lines_index, value):
    lamda = pd.Series({
        (t, year, source, target): value
        for t in timesteps[1:]
        for source, target in zip(
            shared_lines_index['Site In'], shared_lines_index['Site Out']
        )
    })
    lamda.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)
    return lamda


def run_parallel(
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

    (
        timesteps,
        clusters,
        neighbors,
        internal_lines,
        shared_lines,
        shared_lines_index,
        flow_global,
        lamda,
        weighting_order,
    ) = prepare_admm(
        data_all,
        timesteps,
        clusters,
        microgrid_files,
        microgrid_cluster_mode,
        cross_scenario_data,
        noTypicalPeriods,
        hoursPerPeriod,
    )

    n_clusters = len(clusters)

    # store metadata
    metadata = AdmmMetadata(clusters, admmopt)
    with open(join(result_dir, 'metadata.json'), 'w', encoding='utf8') as f:
        json.dump(metadata.to_dict(), f, indent=4)

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
                admmopt,
                n_clusters,
                clusters[ID],
                neighbors[ID],
                internal_lines[ID],
                shared_lines[ID],
                shared_lines_index[ID],
                flow_global[ID],
                lamda[ID],
                queues,
                hoursPerPeriod,
                weighting_order,
                threads,
            )
        )
        procs.append(proc)

    print('Spawning worker processes')

    solver_start = time()
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

    ttime = time()
    solver_time = ttime - solver_start

    admm_objective = sum(result.objective for result in results)

    # print results
    print(f'ADMM solver time: {solver_time:4.0f} s')
    print(f'ADMM objective  : {admm_objective:.4e}')

    return admm_objective


def run_sequential(
    solver_name,
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

    # admm preprocessing
    (
        timesteps,
        clusters,
        neighbors,
        internal_lines,
        shared_lines,
        shared_lines_index,
        flow_global,
        lamda,
        weighting_order,
    ) = prepare_admm(
        data_all,
        timesteps,
        clusters,
        microgrid_files,
        microgrid_cluster_mode,
        cross_scenario_data,
        noTypicalPeriods,
        hoursPerPeriod,
    )

    n_clusters = len(clusters)

    # store metadata (now that all clusters are created)
    metadata = AdmmMetadata(clusters, admmopt)
    with open(join(result_dir, 'metadata.json'), 'w', encoding='utf8') as f:
        json.dump(metadata.to_dict(), f, indent=4)

    # TODO: remove
    # ADMM fields to be indexed by cluster ID. Values are updated in each iteration.
    # obj = [None] * n_clusters
    # flows_all = [None] * n_clusters
    # flows_with_neighbor = [None] * n_clusters
    # primalgap = [None] * n_clusters
    # dualgap = [None] * n_clusters

    models = [
        AdmmModel(
            ID,
            result_dir,
            admmopt,
            neighbors[ID],
            shared_lines[ID],
            shared_lines_index[ID],
            flow_global[ID],
            lamda[ID],
        )
        for ID in range(n_clusters)
    ]

    # place to store models on disk
    model_dir = join(result_dir, 'models')
    os.mkdir(model_dir)

    # measurements
    model_times = []
    pickle_times = []
    unpickle_times = []

    # Create base models, without objective function, and pickle them.
    model_files = []
    for ID in range(n_clusters):
        model_start = time()

        urbs_model = urbs.model.create_model(
            data_all,
            timesteps,
            dt,
            objective,
            sites=clusters[ID],
            shared_lines=shared_lines[ID],
            internal_lines=internal_lines[ID],
            hoursPerPeriod=hoursPerPeriod,
            weighting_order=weighting_order,
        )

        model_time = time() - model_start
        model_times.append(model_time)
        print(f'model_time: {model_time:.2f}')

        # pickle
        model_file = join(model_dir, f'{ID}.pickle')
        model_files.append(model_file)
        with open(model_file, 'wb') as f:
            pickle_start = time()
            pickle.dump(urbs_model, f)
            pickle_time = time() - pickle_start
            pickle_times.append(pickle_time)
            print(f'pickle_time: {pickle_time}')
        del urbs_model

    # unpickle test (TODO: remove)
    for model_file in model_files:
        with open(model_file, 'rb') as f:
            unpickle_start = time()
            urbs_model = pickle.load(f)
            unpickle_time = time() - unpickle_start
            unpickle_times.append(unpickle_time)
            print(f'unpickle_time: {unpickle_time}')
            del urbs_model

    avg_model_time = sum(model_times) / len(model_times)
    print(f'avg_model_time: {avg_model_time:.2f}')

    avg_pickle_time = sum(pickle_times) / len(pickle_times)
    print(f'avg_pickle_time: {avg_pickle_time:.2f}')

    avg_unpickle_time = sum(unpickle_times) / len(unpickle_times)
    print(f'avg_unpickle_time: {avg_unpickle_time:.2f}')

    solver = pyomo.SolverFactory(solver_name)
    solver.set_options(f"LogToConsole=0")
    setup_solver(solver, threads=threads)
    global_convergence = False
    solver_start = time()

    print(f"iter {'primalgap'.ljust(12)} time")

    for nu in range(admmopt.max_iter):
        objectives = []
        for model, model_file in zip(models, model_files):
            with open(model_files[ID], 'rb') as f:
                urbs_model = pickle.load(f)

            objective, _, _, _, _, _ = \
                model.solve_iteration(solver, urbs_model)

            objectives.append(objective)

            del model

        # check convergence
        # TODO: perhaps compute in a centralized fashion
        max_primalgap = max(model.primalgap for model in models)
        # max_mismatch = max(
        #     model.mismatch(k, models[k].flow_global)
        #     for model in models
        #     for k in model.neighbors
        # )

        for ID, model in enumerate(models):
            # TODO: not all fields are needed here
            model.update_flow_global({
                k: AdmmVariableMessage(
                    k,
                    flow = models[k].flows_with_neighbor[ID],
                    lamda = models[k].lamda[
                        models[k].lamda.index.isin(models[k].flows_with_neighbor[ID].index)
                    ],
                    rho = models[k].rho,
                    flow_global = models[k].flow_global.loc[
                        models[k].flow_global.index.isin(
                            models[k].flows_with_neighbor[ID].index
                        )
                    ],
                )
                for k in model.neighbors
            })
            model.update_lamda()
            # TODO: update rho

        iter_time = time() - solver_start
        print(f"{str(nu).rjust(4)} {max_primalgap:.6e} {int(iter_time)}s")

        if max_primalgap < admmopt.primal_tolerance:
            global_convergence = True
            break

    if global_convergence:
        print('Global convergence!')
    else:
        print('Timeout')

    solver_time = time() - solver_start
    admm_objective = sum(objectives)

    print(f'ADMM solver time: {solver_time:4.0f} s')
    print(f'ADMM objective  : {admm_objective:.4e}')

    result = {
        'objective': admm_objective,
        'time': solver_time,
    }

    with open(join(result_dir, 'result.json'), 'w', encoding='utf8') as f:
        json.dump(result, f, indent=4)

    return admm_objective
