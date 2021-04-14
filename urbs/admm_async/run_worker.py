from time import time


def run_worker(ID, s, output):
    """
    ### Args:
    * `ID`: the ordinality of the subproblem
    * `s`: the UrbsAdmmModel instance corresponding to the subproblem
    * `output`: the Queue() object where the results are delivered to
    """

    print("Worker %d initialized successfully!" % (ID,))
    cost_history = []
    max_iter = s.admmopt.max_iter

    for nu in range(max_iter):
        print('Subproblem %d is at iteration %d right now.' % (ID, nu))

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
            print('Subproblem %d at iteration %d solved!. Local cost at %d is: %d. Residprim is: %d'
                  % (ID, nu, ID, cost_history[-1], s.primalgap[-1]))
        print("Time for solving subproblem %d: %ssecs to %ssecs" % (ID, start_time, end_time))

        if converged:
            print("Worker %d converged!" % (ID,))
            break

        s.update_cost_rule()

    print("Local iteration of worker %d is %d" % (ID, nu))
    # save(s.model, os.path.join(s.result_dir, '_{}_'.format(ID),'{}.h5'.format(s.sce)))
    output_package = {
        'cost': cost_history,
        'coupling_flows': s.flow_global,
        'primal_residual': s.primalgap,
        'dual_residual': s.dualgap,
        'received_neighbors': s.received_neighbors,
    }
    output.put((ID - 1, output_package))
