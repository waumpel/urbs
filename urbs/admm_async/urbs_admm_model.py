############################################################################
# This file builds the opf_admm_model class that represents a subproblem
# ADMM algorithm parameters should be defined in AdmmOption
# Package Pypower 5.1.3 is used in this application
############################################################################

from copy import deepcopy
from math import ceil
import queue

import numpy as np
import pandas as pd
import pyomo.environ as pyomo
from pyomo.environ import SolverFactory

from urbs.model import cost_rule_sub

class UrbsAdmmModel(object):
    """
    Encapsulates a local urbs subproblem and implements ADMM steps
    including x-update(solving subproblem), send data to neighbors, receive data
    from neighbors, z-update (global flows) and y-update (lambdas).

    ### Members

    `admmopt`: `AdmmOption` object.
    `dualgap`: List holding the dual gaps after each iteration.
    `flow_global`: `pd.Series` holding the global flow values. Index is
        `['t', 'stf', 'sit', 'sit_']`.
    `flows_all`: `pd.Series` holding the values of the local flow variables after each
        solver iteration. Index is `['t', 'stf', 'sit', 'sit_']`. Initial value is `None`.
    `flows_with_neighbor`: `pd.Series` holding the values of the local flow variables
        with each neighbor after each solver iteration. Index is
        `['t', 'stf', 'sit', 'sit_']`.Initial value is `None`.
    `ID`: ID of this subproblem (zero-based integer).
    `initial_values`: `InitialValues` object.
    `lamda`: `pd.Series` holding the Lagrange multipliers. Index is
        `['t', 'stf', 'sit', 'sit_']`.
    `model`: `pyomo.ConcreteModel`.
    `neighbors`: List of neighbor IDs.
    `n_neighbors`: Number of neighbors.
    `nwait`: Number of updated neighbors required for the next iteration.
    `primalgap`: List of dual gaps after each iteration.
    `received_neighbors`: List holding the number of updated neighbors in each iteration.
    `receiving_queues`: Dict mapping each neighbor ID to a `mp.Queue` for receiving messages
        from that neighbor.
    `recvmsg`: Dict mapping each neighbor ID to the most recent message from that neighbor.
    `result_dir`: Result directory.
    `rho`: Quadratic penalty coefficient. Initial value taken from `initial_values`.
    `scenario_name`: Scenario name.
    `sending_queues`: Dict mapping each neighbor ID to a `mp.Queue` for sending messages to
        that neighbor.
    `shared_lines`: DataFrame of inter-cluster transmission lines. A copy of a slice of the
        'Transmision' DataFrame, enriched with the columns `cluster_from`, `cluster_to` and
        `neighbor_cluster`. Index is
        `['support_timeframe', 'Site In', 'Site Out', 'Transmission', 'Commodity']`.
    `shared_lines_index`: `shared_lines.index.to_frame()`.
    `solver`: `GurobiPersistent` solver interface to `model`.
    `terminated`: Flag indicating whether the solver for this model has terminated, i.e.
        reached convergence or exceeded its maximum number of iterations.
    """

    def __init__(
        self,
        admmopt,
        flow_global,
        ID,
        initial_values,
        lamda,
        neighbors,
        receiving_queues,
        result_dir,
        scenario_name,
        sending_queues,
        shared_lines,
        shared_lines_index,
        model,
    ):
        self.admmopt = admmopt
        self.dualgap = []
        self.flow_global = flow_global
        self.flows_all = None
        self.flows_with_neighbor = None
        self.ID = ID
        self.initial_values = initial_values
        self.lamda = lamda
        self.model = model
        self.neighbors = neighbors
        self.n_neighbors = len(neighbors)
        self.nwait = ceil(self.n_neighbors * admmopt.wait_percent)
        self.primalgap = []
        self.received_neighbors = []
        self.receiving_queues = receiving_queues
        self.recvmsg = {}
        self.result_dir = result_dir
        self.rho = initial_values.rho
        self.scenario_name = scenario_name
        self.sending_queues = sending_queues
        self.shared_lines = shared_lines
        self.shared_lines_index = shared_lines_index

        self.solver = SolverFactory('gurobi_persistent')
        self.solver.set_instance(model, symbolic_solver_labels=False)
        self.solver.set_gurobi_param('Method', 2)
        self.solver.set_gurobi_param('Threads', 1)

        self.terminated = False


    def solve_problem(self):
        self.solver.solve(save_results=False, load_solutions=False, warmstart=True)


    def retrieve_boundary_flows(self):
        """
        Retrieve optimized flow values for shared lines from the solver and store them in
        `self.flows_all` and `self.flows_with_neighbor`.
        """
        index = self.shared_lines_index

        self.solver.load_vars(self.model.e_tra_in[:, :, :, :, :, :])

        flows_all = {}
        flows_with_neighbor = {k: {} for k in self.neighbors}

        for (tm, stf, sit_in, sit_out, tra, com), v in self.model.e_tra_in.items():
            if (sit_in, sit_out) in zip(index['Site In'], index['Site Out']):
                flows_all[(tm, stf, sit_in, sit_out)] = v.value
                k = self.shared_lines.loc[(stf, sit_in, sit_out, tra, com), 'neighbor_cluster']
                flows_with_neighbor[k][(tm, stf, sit_in, sit_out)] = v.value


        flows_all = pd.Series(flows_all)
        flows_all.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)

        for k in flows_with_neighbor:
            flows = pd.Series(flows_with_neighbor[k])
            flows.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)
            flows_with_neighbor[k] = flows

        self.flows_all = flows_all
        self.flows_with_neighbor = flows_with_neighbor


    def active_neighbors(self):
        """
        Return a list of IDs of those neighbors who have not terminated yet.
        """
        return [
            k for k in self.neighbors
            if k not in self.recvmsg or not self.recvmsg[k].terminated
        ]


    def recv(self, block):
        """
        Check for new messages from active neighbors and store them in `self.recvmsg`.

        If `block` is true, wait for messages from at least `self.n_wait` neighbors
        (or fewer if not enough neighbors remain active).
        If `block` is false, perform at most `self.admmopt.pollrounds` pollrounds.

        Return the number of updated neighbors and append that number to
        `self.received_neighbors`.
        """
        active_neighbors = self.active_neighbors()
        n_wait = min(self.nwait, len(active_neighbors))
        twait = self.admmopt.poll_wait_time
        pollrounds = self.admmopt.pollrounds
        pollround = 0
        new_msgs = {}

        while len(new_msgs) < n_wait:
            # read accumulated messages from active neighbors
            for k in active_neighbors:
                que = self.receiving_queues[k]
                while not que.empty():
                    new_msgs[k] = que.get(block=False) # don't wait

            # otherwise, wait for a message from the last neighbor
            k = active_neighbors[-1]
            que = self.receiving_queues[k]
            try:
                new_msgs[k] = que.get(timeout=twait)
            except queue.Empty:
                pass

            pollround += 1
            # break if non-blocking and pollrounds exceeded
            if not block and pollround >= pollrounds:
                break

        # store new messages
        self.recvmsg.update(new_msgs)

        # store and return number of received neighbors
        nrecv = len(new_msgs)
        self.received_neighbors.append(nrecv)
        return nrecv


    def send(self):
        """
        Send an `AdmmMessage` with the current status to each neighbor.
        """
        for k, que in self.sending_queues.items():
            msg = AdmmMessage(
                self.ID,
                k,
                self.flows_with_neighbor[k],
                self.rho,
                self.lamda[self.lamda.index.isin(self.flows_with_neighbor[k].index)],
                self.primalgap[-1],
                self.terminated
            )
            que.put(msg)


    def update_flow_global(self):
        """
        Update `self.flow_global` for ALL neighbors, using the values from the most recent
        messages. If a neighbor hasn't sent any messages yet, the `initial_values` are used.
        Also calculate the new dual gap and append it to `self.dualgap`.
        """
        flow_global_old = deepcopy(self.flow_global)
        for k in self.neighbors:
            if k in self.recvmsg:
                msg = self.recvmsg[k]
                lamda = msg.lamda
                flow = msg.flow
                rho = msg.rho
            else:
                lamda = self.initial_values.lamda
                flow = self.initial_values.flow
                rho = self.initial_values.rho

            self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)] = (
                (self.lamda.loc[self.lamda.index.isin(self.flows_with_neighbor[k].index)] +
                 lamda + self.flows_with_neighbor[k] * self.rho + flow * rho) /
                (self.rho + rho))

        self.dualgap.append(
            self.rho * np.sqrt(
                np.square(self.flow_global - flow_global_old).sum(axis=0)
            )
        )


    def update_lamda(self):
        """
        Update `self.lamda` using the updated values of `self.flows_all` and
        `self.flow_global`.
        """
        self.lamda = self.lamda + self.rho * (self.flows_all - self.flow_global)


    def update_rho(self, nu):
        """
        Calculate the new primal gap, append it to `self.primalgap`.
        Update `self.rho` according to the new primal and dual gaps unless the
        current iteration is above `self.admmopt.rho_update_nu`.

        ### Arguments
        * `nu`: The current iteration.
        """
        # primal gap normalized by number of global constraints
        primalgap = np.sqrt(np.square(self.flows_all - self.flow_global).sum(axis=0)) \
            / min(1, len(self.flow_global))
        self.primalgap.append(primalgap)

        if nu <= self.admmopt.rho_update_nu:
            if primalgap > self.admmopt.mu * self.dualgap[-1]:
                self.rho = min(self.admmopt.rho_max, self.rho * self.admmopt.tau)
            elif self.dualgap[-1] > self.admmopt.mu * primalgap:
                self.rho = min(self.rho / self.admmopt.tau, self.admmopt.rho_max)


    def update_cost_rule(self):
        """
        Update those components of `self.model` that use `cost_rule_sub` to reflect
        changes to `self.flow_global`, `self.lamda` and `self.rho`.
        Currently only supports models with `cost` objective, i.e. only the objective
        function is updated.
        """
        m = self.model
        if m.obj.value == 'cost':
            m.del_component(m.objective_function)
            m.objective_function = pyomo.Objective(
                rule=cost_rule_sub(flow_global=self.flow_global,
                                   lamda=self.lamda,
                                   rho=self.rho),
                sense=pyomo.minimize,
                doc='minimize(cost = sum of all cost types)')

            self.solver.set_objective(m.objective_function)
        else:
            raise NotImplementedError("Objectives other than 'cost' are not supported.")


    def choose_max_rho(self):
        """
        Set `self.rho` to the maximum rho value among self and neighbors.
        """
        self.rho = max(self.rho, *[msg.rho for msg in self.recvmsg.values()])


    def is_converged(self):
        """
        Return whether the current primal gap is below the tolerance threshold.
        """
        return self.primalgap[-1] < self.admmopt.primal_tolerance


# ##--------ADMM parameters specification -------------------------------------
class AdmmOption(object):
    """
    This class defines all the parameters to use in ADMM.
    """
    # TODO: docstring

    def __init__(self):
        self.rho_max = 10  # upper bound for penalty rho
        self.tau_max = 1.5  # parameter for residual balancing of rho
        self.tau = 1.05  # multiplier for increasing rho
        self.zeta = 1  # parameter for residual balancing of rho
        self.theta = 0.99  # multiplier for determining whether to update rho
        self.mu = 10  # multiplier for determining whether to update rho
        self.pollrounds = 5
        self.poll_wait_time = 0.001  # waiting time of receiving from one pipe
        self.wait_percent = 0.2  # waiting percentage of neighbors (0, 1]
        self.max_iter = 20  # local maximum iteration
        self.rho_update_nu = 50 # rho is updated only for the first 50 iterations
        self.primal_tolerance = 0.1 # the relative convergece tolerance, to be multiplied with len(s.flow_global)


class AdmmMessage(object):
    """
    This class defines the message region i sends to/receives from j.
    """
    # TODO: docstring

    def __init__(self, source, target, flow, rho, lamda, primalgap, terminated):
        self.source = source  # source region ID
        self.target = target  # destination region ID
        self.flow = flow
        self.rho = rho
        self.lamda = lamda
        self.primalgap = primalgap
        self.terminated = terminated
