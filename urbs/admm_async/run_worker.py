from time import sleep, time

import pandas as pd

import urbs.model
from .urbs_admm_model import AdmmStatus, UrbsAdmmModel
from . import input_output

def log_generator(ID, logqueue):
    """
    Return a log function that prefixes messages with the process `ID`, prints them to
    stdout, and sends them to the `logqueue`.
    """
    prefix = f'Process[{ID}] '
    def fun(*args):
        msg = prefix + f'{" ".join(str(arg) for arg in args)}'
        print(msg)
        logqueue.put(msg)
    return fun


def create_model(
    ID, # TODO: remove
    data_all,
    scenario_name,
    timesteps,
    year,
    initial_values,
    admmopt,
    n_clusters,
    sites,
    neighbors,
    shared_lines,
    internal_lines,
    cluster_from,
    cluster_to,
    neighbor_cluster,
    queues,
    result_dir,
    ):

    print(f'Creating model {ID} for sites {sites}') # TODO: remove
    index = shared_lines.index.to_frame()

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

    model = urbs.model.create_model(data_all, timesteps, type='sub',
                        sites=sites,
                        data_transmission_boun=shared_lines,
                        data_transmission_int=internal_lines,
                        flow_global=flow_global,
                        lamda=lamda,
                        rho=admmopt.rho,
                        ID=ID) # TODO: remove ID parameter

    # enlarge shared_lines (copies of slices of data_all['transmission'])
    shared_lines['cluster_from'] = cluster_from
    shared_lines['cluster_to'] = cluster_to
    shared_lines['neighbor_cluster'] = neighbor_cluster

    return UrbsAdmmModel(
        admmopt = admmopt,
        flow_global = flow_global,
        ID = ID,
        lamda = lamda,
        model = model,
        n_clusters = n_clusters,
        neighbors = neighbors,
        queues = queues,
        regions = sites,
        result_dir = result_dir,
        scenario_name = scenario_name,
        shared_lines = shared_lines,
        shared_lines_index = index,
    )


def run_worker(
    s, # TODO: remove
    ID,
    data_all,
    scenario_name,
    timesteps,
    year,
    initial_values,
    admmopt,
    n_clusters,
    sites,
    neighbors,
    shared_lines,
    internal_lines,
    cluster_from,
    cluster_to,
    neighbor_cluster,
    queues,
    result_dir,
    output,
    logqueue
    ):
    """
    Main function for child processes of ADMM. Iteratively solves one subproblem of ADMM.

    ### Args:
    * `s`: `UrbsAdmmModel` representing the subproblem.
    * `output`: `multiprocessing.Queue` for sending results.
    * `logqueue`: `mp.Queue` for sending log messages. These are written to a shared log
                  file by the master process.
    """

    # TODO: switch back to parallel model creation
    # s = create_model(
    #     ID,
    #     data_all,
    #     scenario_name,
    #     timesteps,
    #     year,
    #     initial_values,
    #     admmopt,
    #     n_clusters,
    #     sites,
    #     neighbors,
    #     shared_lines,
    #     internal_lines,
    #     cluster_from,
    #     cluster_to,
    #     neighbor_cluster,
    #     queues,
    #     result_dir,
    # )

    max_iter = s.admmopt.max_iter
    solver_times = [] # Stores the duration of each solver iteration
    timestamps = [] # Stores the times after each solver iteration

    log = log_generator(s.ID, logqueue)
    log(f'Starting subproblem for regions {", ".join(s.regions)}.')

    for nu in range(max_iter):

        if nu % 10 == 0:
            log(f'Iteration {nu}')

        start = time()
        s.solve_problem()
        solver_time = time() - start
        solver_times.append(solver_time)

        s.retrieve_boundary_flows()
        s.update_primalgap()

        # Take the timestamp now, when objective and primal gap are known for this iteration.
        timestamps.append(time())

        s.receive()
        if s.terminated():
            log('Received termination msg: Terminating.')
            break

        if s.status_update:
            s.send_status()

        s.send_variables()

        if s.all_global_convergence():
            log(f'Global convergence at iteration {nu}!')
            break

        msg_counter = 0

        while len(s.updated[-1]) < s.n_wait or (s.all_converged()):

            if msg_counter > 1000:
                log('Timeout while checking for messages: Terminating.')
                s.set_status(AdmmStatus.TERMINATED)
                s.send_status()
                break

            sleep(s.admmopt.wait_time)
            senders = s.receive()

            if s.terminated():
                log('Received termination msg: Terminating.')
                break
            if not senders:
                continue
            if s.status_update:
                s.send_status()
            if s.all_global_convergence():
                break

            msg_counter += 1

        if s.all_global_convergence():
            log(f'Global convergence at iteration {nu}!')
            break

        if s.terminated():
            break

        if nu == max_iter - 1:
            log('Timeout: Terminating.')
            s.set_status(AdmmStatus.TERMINATED)
            s.send_status()
            break

        s.update_lamda()
        s.update_flow_global()
        s.update_rho()

        s.update_cost_rule()

    # save(s.model, os.path.join(s.result_dir, '_{}_'.format(ID),'{}.h5'.format(s.sce)))
    cluster_results = input_output.cluster_results_dict(
        s.ID,
        s.regions,
        s.flow_global.tolist(),
        timestamps,
        s.objective_values,
        s.primalgaps,
        s.dualgaps,
        s.max_mismatch_gaps,
        s.rhos,
        s.raw_dualgaps,
    )
    output.put(cluster_results)
