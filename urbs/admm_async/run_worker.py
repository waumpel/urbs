from time import sleep, time


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


def run_worker(s, output, logqueue, iteration_results):
    """
    Main function for child processes of ADMM. Iteratively solves one subproblem of ADMM.

    ### Args:
    * `s`: `UrbsAdmmModel` representing the subproblem.
    * `output`: `multiprocessing.Queue` for sending results.
    * `logqueue`: `mp.Queue` for sending log messages. These are written to a shared log
                  file by the master process.
    """
    max_iter = s.admmopt.max_iter
    solver_times = []

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

        iteration_results.put({
            'sender': s.ID,
            'time': time(),
            'primalgap': s.primalgaps[-1],
            'dualgap': s.dualgaps[-1],
        })

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
        # s.choose_max_rho() # TODO: not needed?
        # s.update_rho(nu) # TODO

        s.update_cost_rule()

    # save(s.model, os.path.join(s.result_dir, '_{}_'.format(ID),'{}.h5'.format(s.sce)))
    output_package = {
        'cost': s.objective_values,
        'coupling_flows': s.flow_global,
        'primal_residual': s.primalgaps,
        'dual_residual': s.dualgaps,
    }
    output.put((s.ID, output_package))
