from time import time


def safe_print(lock, *args):
    """
    Print to `stdout` in a synchronized fashion.

    ### Arguments
    * `lock`: `threading.Lock` that controls the synchronized printing.
    * `*args`: What to print. Same as the `*values` argument of `print`.
    """
    lock.acquire()
    try:
        print(*args)
    finally:
        lock.release()


def run_worker(s, output, printlock):
    """
    Main function for child processes of ADMM. Iteratively solves one subproblem of ADMM.

    ### Args:
    * `s`: `UrbsAdmmModel` representing the subproblem.
    * `output`: `multiprocessing.Queue` for sending results.
    * `printlock`: `threading.Lock` for synchronized printing.
    """

    ID = s.ID + 1 # one-based integer for printing
    safe_print(printlock, "Worker %d initialized successfully!" % (ID,))
    cost_history = []
    max_iter = s.admmopt.max_iter

    for nu in range(max_iter):
        safe_print(printlock, 'Subproblem %d is at iteration %d right now.' % (ID, nu))

        start_time = time()
        s.solve_problem()
        end_time = time()

        s.retrieve_boundary_flows()
        cost_history.append(s.solver._solver_model.objval) # TODO: use public method instead

        # Don't block in the first iteration to avoid deadlock.
        s.recv(block=nu > 0)

        s.update_flow_global()
        # s.choose_max_rho() # TODO: not needed?
        s.update_lamda()
        s.update_rho(nu)

        converged = s.is_converged()
        last_iteration = nu == max_iter - 1
        s.terminated = converged or last_iteration

        s.send()

        if nu % 1 == 0:
            safe_print(printlock, 'Subproblem %d at iteration %d solved!. Local cost at %d is: %d. Residprim is: %d'
                % (ID, nu, ID, cost_history[-1], s.primalgap[-1]))
        safe_print(printlock, "Time for solving subproblem %d: %ssecs to %ssecs" % (ID, start_time, end_time))

        if converged:
            safe_print(printlock, "Worker %d converged!" % (ID,))
            break

        s.update_cost_rule()

    safe_print(printlock, "Local iteration of worker %d is %d" % (ID, nu))
    # save(s.model, os.path.join(s.result_dir, '_{}_'.format(ID),'{}.h5'.format(s.sce)))
    output_package = {
        'cost': cost_history,
        'coupling_flows': s.flow_global,
        'primal_residual': s.primalgap,
        'dual_residual': s.dualgap,
        'received_neighbors': s.received_neighbors,
    }
    output.put((ID - 1, output_package))
