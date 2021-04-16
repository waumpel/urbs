from time import time


def safe_print(ID, lock, *args):
    """
    Print to `stdout` in a synchronized fashion.

    ### Arguments
    * `ID`: ID of the printing process (one-based integer).
    * `lock`: `threading.Lock` that controls the synchronized printing.
    * `*args`: What to print. Same as the `*values` argument of `print`.
    """
    lock.acquire()
    try:
        print(f'Process[{ID}]', *args)
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
    cost_history = []
    max_iter = s.admmopt.max_iter
    received = 0
    solver_times = []

    safe_print(ID, printlock, f'Starting subproblem for regions {", ".join(s.regions)}.')

    for nu in range(max_iter):
        safe_print(ID, printlock, f'Iteration {nu}')
        safe_print(ID, printlock, f'Starting with {received} updated neighbors')

        start = time()
        s.solve_problem()
        solver_time = time() - start
        solver_times.append(solver_time)
        safe_print(ID, printlock, f'Solved in {solver_time:.2f} seconds')

        cost_history.append(s.solver._solver_model.objval) # TODO: use public method instead

        s.retrieve_boundary_flows()

        # Don't block in the first iteration to avoid deadlock.
        received = s.recv(block=nu > 0)

        s.update_flow_global()
        # s.choose_max_rho() # TODO: not needed?
        s.update_lamda()
        s.update_rho(nu)

        converged = s.is_converged()
        last_iteration = nu == max_iter - 1
        s.terminated = converged or last_iteration

        s.send()

        if nu % 1 == 0:
            safe_print(ID, printlock, f'Primal gap: {s.primalgap[-1]:.4e}')

        if converged:
            safe_print(ID, printlock, 'Converged!')
            break

        s.update_cost_rule()

    # save(s.model, os.path.join(s.result_dir, '_{}_'.format(ID),'{}.h5'.format(s.sce)))
    output_package = {
        'cost': cost_history,
        'coupling_flows': s.flow_global,
        'primal_residual': s.primalgap,
        'dual_residual': s.dualgap,
    }
    output.put((s.ID, output_package))
