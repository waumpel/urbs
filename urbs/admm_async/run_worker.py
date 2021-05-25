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


def run_worker(s, output, logqueue):
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
    converged = False

    log = log_generator(s.ID, logqueue)
    log(f'Starting subproblem for regions {", ".join(s.regions)}.')

    for nu in range(max_iter):
        celebration = False
        if nu % 10 == 0:
            log(f'Iteration {nu}')

        start = time()
        s.solve_problem()
        solver_time = time() - start
        solver_times.append(solver_time)

        s.retrieve_boundary_flows()
        s.calc_primalgap()
        if s.converged:
            log(f'Converged at iteration {nu}')
            celebration = True

        s.receive()
        s.send()
        converged = s.global_convergence()

        if not converged and not s.terminated:
            while len(s.received[-1]) < s.n_wait or s.converged:

                sleep(s.admmopt.wait_time)
                s.receive()

                s.converged = s.is_converged()
                converged = s.global_convergence()

                # In case of global convergence or termination, send another msg so that
                # other processes are notified.
                # Having this check inside the outer if-statement avoids potentially sending
                # two messages.
                if converged or s.terminated:
                    s.send()
                    break

                # In case of local convergence, send updates to the iteration table to the
                # neighbors, so that global convergence can be detected.
                if s.converged:
                    if s.update_flag: # TODO: s.all_converged() instead of s.converged sufficient?
                        if not celebration:
                            log(f'Converged at iteration {nu}')
                            celebration = True
                        s.send()
                elif celebration:
                    log('No longer converged')
                    celebration = False



        if converged:
            log(f'Global convergence at iteration {nu}!')
            break

        if s.terminated:
            log('Received termination msg: Terminating.')
            break

        if nu == max_iter - 1:
            log('Timeout: Terminating.')
            s.terminated = True
            s.send()
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
