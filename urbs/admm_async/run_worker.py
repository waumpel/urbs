from time import sleep, time

import pandas as pd

from urbs.model import create_model
from .urbs_admm_model import UrbsAdmmModel

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


def run_worker(
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

    model = create_model(data_all, timesteps, type='sub',
                        sites=sites,
                        data_transmission_boun=shared_lines,
                        data_transmission_int=internal_lines,
                        flow_global=flow_global,
                        lamda=lamda,
                        rho=admmopt.rho)

    sending_queues = {
        target: queues[target] for target in neighbors
    }

    # enlarge shared_lines (copies of slices of data_all['transmission'])
    shared_lines['cluster_from'] = cluster_from
    shared_lines['cluster_to'] = cluster_to
    shared_lines['neighbor_cluster'] = neighbor_cluster

    s = UrbsAdmmModel(
        admmopt = admmopt,
        flow_global = flow_global,
        ID = ID,
        lamda = lamda,
        model = model,
        n_clusters = n_clusters,
        neighbors = neighbors,
        receiving_queue = queues[ID],
        regions = sites,
        result_dir = result_dir,
        scenario_name = scenario_name,
        sending_queues = sending_queues,
        shared_lines = shared_lines,
        shared_lines_index = index,
    )

    max_iter = s.admmopt.max_iter
    solver_times = [] # Stores the duration of each solver iteration
    timestamps = [] # Stores the times after each solver iteration

    log = log_generator(s.ID, logqueue)
    log(f'Starting subproblem for regions {", ".join(s.regions)}.')

    for nu in range(max_iter):
        # Flag indicating whether current convergence status has been printed
        celebration = False

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

        if s.local_convergence():
            log(f'Converged at iteration {nu}')
            celebration = True

        s.receive()
        s.send()

        if not s.global_convergence() and not s.terminated:
            while len(s.updated[-1]) < s.n_wait or (s.all_converged()):

                sleep(s.admmopt.wait_time)
                full, status = s.receive()
                if not (full or status):
                    continue

                # In case of global convergence or termination, send another msg so that
                # other processes are notified.
                # Having this check inside the outer if-statement avoids potentially sending
                # two messages.
                if s.global_convergence() or s.terminated:
                    s.send_status()
                    break

                # If `s.all_converged() == True`, this cluster may not reiterate, so any
                # updates to `s.status` must be sent to the neighbors.
                # (Otherwise, global convergence may not be detected.)
                if s.all_converged():
                    if s.status_update:
                        if not celebration:
                            log(f'Converged at iteration {nu}')
                            celebration = True
                        s.send_status()
                # No need to send another msg; this cluster will either reiterate or
                # reach convergence once again.
                elif celebration:
                    log('No longer converged')
                    celebration = False

        if s.global_convergence():
            log(f'Global convergence at iteration {nu}!')
            break

        if s.terminated:
            log('Received termination msg: Terminating.')
            break

        if nu == max_iter - 1:
            log('Timeout: Terminating.')
            s.terminated = True
            s.send_status()
            break

        s.update_lamda()
        s.update_flow_global()
        s.update_rho()
        s.choose_max_rho()

        s.update_cost_rule()

    # save(s.model, os.path.join(s.result_dir, '_{}_'.format(ID),'{}.h5'.format(s.sce)))
    output.put({
        'ID': s.ID,
        'regions': s.regions,
        'timestamps': timestamps,
        'objective': s.objective_values,
        'primal_residual': s.primalgaps,
        'dual_residual': s.dualgaps,
        'coupling_flows': s.flow_global.tolist(),
    })
