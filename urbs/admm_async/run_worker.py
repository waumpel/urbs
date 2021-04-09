from time import time

import numpy as np
from pyomo.environ import SolverFactory


def run_worker(ID, s, output):
    """

    Args:
        ID: the ordinality of the subproblem
        s: the UrbsAdmmModel instance corresponding to the subproblem
        output: the Queue() object where the results are delivered to

    """
    s.sub_persistent = SolverFactory('gurobi_persistent')
    s.sub_persistent.set_instance(s.sub_pyomo, symbolic_solver_labels=False)
    s.sub_persistent.set_gurobi_param('Method', 2)
    s.sub_persistent.set_gurobi_param('Threads', 1)
    s.neighbor_clusters = s.shared_lines.neighbor_cluster.unique()

    print("Worker %d initialized successfully!" % (ID,))
    maxit = s.admmopt.iterMaxlocal # get maximum iteration
    s.gapAll = [10 ** 8] * s.na
    cost_history = []
    s.convergetol = s.admmopt.conv_rel * (len(s.flow_global)+1) # convergence criteria for maximum primal gap # TODO

    for nu in range(maxit):
        print('Subproblem %d is at iteration %d right now.' % (ID, nu))

        s.fix_flow_global()
        s.fix_lamda()

        if nu > 0:
            s.set_quad_cost(rho_old)

        start_time = time()
        s.solve_problem()
        end_time = time()

        s.retrieve_boundary_flows()
        cost_history.append(s.sub_persistent._solver_model.objval) # TODO: use public method instead
        rho_old = s.rho

        s.recv(pollrounds=5)

        if s.recvmsg:
            # TODO: what if no neighbor updated? record dualgap, what gets updated?
            # rho? what if we're above rho_update_nu?
            s.update_flow_global()
            # s.choose_max_rho() # TODO: not needed?
            s.update_lamda()
            s.update_rho(nu)

        s.send()

        if nu % 1 == 0:
            print('Subproblem %d at iteration %d solved!. Local cost at %d is: %d. Residprim is: %d'
                  % (ID, nu, ID, cost_history[-1], s.primalgap[-1]))
        print("Time for solving subproblem %d: %ssecs to %ssecs" % (ID, start_time, end_time))

        if s.is_converged():
            print("Worker %d converged!" % (ID,))
            break

    print("Local iteration of worker %d is %d" % (ID, nu))
    # save(s.sub_pyomo, os.path.join(s.result_dir, '_{}_'.format(ID),'{}.h5'.format(s.sce)))
    output_package = {
        'cost': cost_history,
        'coupling_flows': s.flow_global,
        'primal_residual': s.primalgap,
        'dual_residual': s.dualgap,
        'received_neighbors': s.received_neighbors,
    }
    output.put((ID - 1, output_package))
