############################################################################
# This file builds the opf_admm_model class that represents a subproblem
# ADMM algorithm parameters should be defined in AdmmOption
# Package Pypower 5.1.3 is used in this application
############################################################################

from copy import deepcopy
from math import ceil
from os.path import join

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
    `dualgaps`: List holding the dual gaps in each iteration (starts with a single `0`
        entry).
    `flow_global`: `pd.Series` holding the global flow values. Index is
        `['t', 'stf', 'sit', 'sit_']`.
    `flows_all`: `pd.Series` holding the values of the local flow variables after each
        solver iteration. Index is `['t', 'stf', 'sit', 'sit_']`. Initial value is `None`.
    `flows_with_neighbor`: `pd.Series` holding the values of the local flow variables
        with each neighbor after each solver iteration. Index is
        `['t', 'stf', 'sit', 'sit_']`.Initial value is `None`.
    `ID`: ID of this subproblem (zero-based integer).
    `lamda`: `pd.Series` holding the Lagrange multipliers. Index is
        `['t', 'stf', 'sit', 'sit_']`.
    `logfile`: Logfile for this cluster.
    `max_mismatch_gaps`: List holding the maximal constraint mismatch in each iteration.
    `messages`: Dict mapping each neighbor ID to the most recent message from that neighbor.
    `mismatch_convergence`: Dict mapping each neighbor ID to a flag indicating the current
        mismatch convergence status.
    `model`: `pyomo.ConcreteModel`.
    `n_clusters`: Total number of clusters.
    `n_neighbors`: Number of neighbors.
    `n_wait`: Number of updated neighbors required for the next iteration.
    `neighbor_queues`: Dict mapping neighbor IDs to `mp.Queue`s for sending messages
        to that neighbor.
    `neighbors`: List of neighbor IDs.
    `nu`: Current iteration.
    `objective_values`: List of objective function values after each iteration.
    `primalgaps`: List of primal gaps after each iteration.
    `receiving_queue`: `mp.Queue` for receiving messages from neighbors.
    `regions`: List of region names in this subproblem.
    `result_dir`: Result directory.
    `rho`: Quadratic penalty coefficient.
    `rhos`: List holding the penalty parameter of each iteration.
    `scenario_name`: Scenario name.
    `sending_queues`: Dict mapping each cluster ID (except `self.ID`) to a `mp.Queue` for
        sending messages to that cluster.
    `shared_lines`: `pd.DataFrame` of inter-cluster transmission lines. A copy of a slice of
        the 'Transmision' DataFrame, enriched with the columns `cluster_from`, `cluster_to`
        and `neighbor_cluster`. Index is
        `['support_timeframe', 'Site In', 'Site Out', 'Transmission', 'Commodity']`.
    `shared_lines_index`: `shared_lines.index.to_frame()`.
    `solver`: `GurobiPersistent` solver interface to `model`.
    `status`: List of lists, each holding the currently known status of one cluster,
        as received through messages. Each list has three entries: iteration counter,
        local convergence flag, and consensus flag.
    `status_update`: Flag indicating whether `self.status` has been updated since
        the last time a status message has been sent.
    `terminated`: Flag indicating whether the solver for this model has terminated.
    `updated`: List of sets holding the neighbors who have sent new variables before
        the start of each iteration (starts with an empty set as the single entry).
    """

    def __init__(
        self,
        admmopt,
        flow_global,
        ID,
        lamda,
        model,
        n_clusters,
        neighbors,
        queues,
        regions,
        result_dir,
        scenario_name,
        shared_lines,
        shared_lines_index,
    ):
        self.admmopt = admmopt
        self.dualgaps = [0]
        self.flow_global = flow_global
        self.flows_all = None
        self.flows_with_neighbor = None
        self.ID = ID
        self.lamda = lamda
        self.logfile = open(join(result_dir, f'process-{ID}.log'), 'w', encoding='utf8')
        self.max_mismatch_gaps = []
        self.messages = {}
        self.mismatch_convergence = {k: False for k in neighbors}
        self.model = model
        self.n_clusters = n_clusters
        self.n_neighbors = len(neighbors)
        self.n_wait = ceil(self.n_neighbors * admmopt.wait_percent)
        self.neighbor_queues = {k: queues[k] for k in neighbors}
        self.neighbors = neighbors
        self.nu = -1
        self.objective_values = []
        self.primalgaps = []
        self.receiving_queue = queues[ID]
        self.regions = regions
        self.result_dir = result_dir
        self.rho = admmopt.rho
        self.rhos = []
        self.scenario_name = scenario_name
        self.sending_queues = {i: queues[i] for i in range(len(queues)) if i != ID}
        self.shared_lines = shared_lines
        self.shared_lines_index = shared_lines_index

        self.solver = SolverFactory('gurobi_persistent')
        self.solver.set_instance(model, symbolic_solver_labels=False)
        self.solver.set_gurobi_param('Method', 2)
        self.solver.set_gurobi_param('Threads', 1)

        self.status = [False] * n_clusters
        self.status_update = False
        self.terminated = False
        self.updated = [set()]


    def __del__(self):
        self.logfile.close()


    def log(self, s):
        self.logfile.write(s + '\n')
        self.logfile.flush()


    def solve_problem(self):
        """
        Start a new iteration and solve the optimization problem.
        """
        self.nu += 1

        self.log('---------------')
        self.log(f'Iteration {self.nu}')
        self.log('---------------')
        self.log(f'Starting with {len(self.updated[-1])} updated neighbors: {self.updated[-1]}')

        self.updated.append(set())
        self.max_mismatch_gaps.append(0)
        self.rhos.append(self.rho)

        self.solver.solve(save_results=False, load_solutions=False, warmstart=True)
        self.objective_values.append(self.solver._solver_model.objval) # TODO: use public method instead

        self.log(f'Objective value: {self.objective_values[-1]}')


    def send_variables(self):
        """
        Send an `AdmmMessage` with the current variables to all neighbors.
        """
        self.log('Sending variable message')

        for k, que in self.neighbor_queues.items():
            msg = AdmmMessage(
                sender = self.ID,
                flow = self.flows_with_neighbor[k],
                lamda = self.lamda[self.lamda.index.isin(self.flows_with_neighbor[k].index)],
                rho = self.rho,
                flow_global = self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)],
            )
            que.put(msg)


    def send_status(self):
        """
        Send an `AdmmStatusMessage` with the current status to all other clusters.
        """
        self.log(f'Sending status message: converged: {self.local_convergence()} terminated: {self.terminated}')

        msg = AdmmStatusMessage(
            sender = self.ID,
            converged = self.local_convergence(),
            terminated = self.terminated
        )
        for que in self.sending_queues.values():
            que.put(msg)

        self.status_update = False


    def receive(self):
        """
        Check for new messages from all clusters.

        Check for termination signals, update status and mismatch convergence.

        Return a set of all clusters that have sent a message.
        """
        self.log('Checking for new messages...')

        variable_senders = set()
        status_msgs = {}

        # read accumulated messages from active neighbors
        while not self.receiving_queue.empty():
            msg = self.receiving_queue.get(block=False)
            if isinstance(msg, AdmmMessage):
                self.messages[msg.sender] = msg
                variable_senders.add(msg.sender)
            elif isinstance(msg, AdmmStatusMessage):
                status_msgs[msg.sender] = msg
                if msg.terminated:
                    self.log('Received msg from a terminated process')
                    self.terminated = True
                    return
            else:
                raise RuntimeError(f'Received a msg of unrecognized type: {type(msg)}')

        senders = variable_senders.union(status_msgs.keys())
        if senders:
            self.log(f'New messages from {len(senders)} clusters: {senders}')
            self.log(f'Thereof variable updates: {variable_senders}')
        else:
            self.log('No new messages.')

        self.updated[-1] = self.updated[-1].union(variable_senders)
        if variable_senders:
            self.update_mismatch(variable_senders)

        for k, msg in status_msgs.items():
            self.log(f'Cluster {k} has converged' if msg.converged else
                     f'Cluster {k} is no longer converged')
            self.status[k] = msg.converged
            self.log(f'New status: {self.status}')

        return senders


    def update_dualgap(self, flow_global_old):
        """
        Calculate the new dual gap.
        """
        self.log('Updating dual gap')
        if self.admmopt.tolerance_mode == 'absolute':
            dualgap = (self.rho * np.square(self.flow_global - flow_global_old).sum(axis=0) /
                       min(1, len(self.flow_global)))
        elif self.admmopt.tolerance_mode == 'relative':
            dualgap = (self.rho * np.square(self.flow_global - flow_global_old).sum(axis=0) /
                       max(1, np.square(flow_global_old).sum(axis=0)))
        self.dualgaps.append(dualgap)
        # No need to call `update_convergence` here; this is done in the next call to
        # `update_primalgap`.


    def update_primalgap(self):
        """
        Calculate the new primal gap and call `self.update_convergence`.
        """
        self.log('Updating primal gap')
        if self.admmopt.tolerance_mode == 'absolute':
            primalgap = (np.square(self.flows_all - self.flow_global).sum(axis=0) /
                         min(1, len(self.flow_global)))

        elif self.admmopt.tolerance_mode == 'relative':
            primalgap = (np.square(self.flows_all - self.flow_global).sum(axis=0) /
                         max(1, np.square(self.flow_global).sum(axis=0)))

        self.log(f'Primal gap: {primalgap}')
        self.primalgaps.append(primalgap)
        self.update_convergence()


    def update_mismatch(self, senders):
        """
        Update the mismatch convergence status with `senders` to reflect the information
        received in new messages.

        If mismatch convergence with all neighbors is gained or lost, call
        `self.update_convergence()`.
        """
        self.log('Checking mismatch convergence...')
        old_value = all(self.mismatch_convergence.values())
        gaps = []

        for k in senders:
            mismatch_gap = self.calc_mismatch_gap(k)
            gaps.append(mismatch_gap)
            self.mismatch_convergence[k] = mismatch_gap < self.admmopt.mismatch_tolerance

        self.log(f'Current mismatch status: {self.mismatch_convergence}')

        self.max_mismatch_gaps[-1] = max(self.max_mismatch_gaps[-1], float(max(gaps)))

        new_value = all(self.mismatch_convergence.values())
        if new_value != old_value:
            self.update_convergence()


    def calc_mismatch_gap(self, k):
        """
        Calculate the current mismatch gap with neighbor `k`.
        """
        msg = self.messages[k]
        flow_global_with_k = self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)]

        if self.admmopt.tolerance_mode == 'absolute':
            mismatch_gap = (np.square(flow_global_with_k - msg.flow_global).sum(axis=0) /
                            min(1, len(self.flow_global)))
        elif self.admmopt.tolerance_mode == 'relative':
            mismatch_gap = (np.square(flow_global_with_k - msg.flow_global).sum(axis=0) /
                            max(1, np.square(flow_global_with_k).sum(axis=0),
                                np.square(msg.flow_global).sum(axis=0)))

        return mismatch_gap


    def update_convergence(self):
        """
        Check for local convergence, i.e. if primal gap, dual gap, and constraint mismatch
        with all neighbors are below their respective tolerance thresholds.

        Update the value of `self.status[self.ID]` and set `self.status_update` if the value
        was modified.

        Return the new value.
        """
        self.log('Checking convergence...')
        old_value = self.local_convergence()
        primal = self.primalgaps[-1] < self.admmopt.primal_tolerance
        dual = self.dualgaps[-1] < self.admmopt.dual_tolerance
        mismatch = all(self.mismatch_convergence.values())
        self.log(f'primal: {primal} dual: {dual} mismatch: {mismatch}')
        new_value = primal and dual and mismatch

        if new_value == old_value:
            self.log('Still converged' if new_value else 'Still not converged')
        else:
            self.log('Reached convergence' if new_value else 'No longer converged')
            self.status[self.ID] = new_value
            self.status_update = True

        return new_value


    def update_rho(self):
        """
        Increase `self.rho` if the primal gap does not decrease sufficiently.
        """
        if self.admmopt.penalty_mode == 'increasing':
            if (self.nu > 0 and
                self.primalgaps[-1] > self.admmopt.primal_decrease * self.primalgaps[-2]):
                self.rho = min(self.admmopt.max_penalty, self.rho * self.admmopt.penalty_mult)

        elif self.admmopt.penalty_mode == 'residual_balancing':
            if self.nu > 0:
                if self.primalgaps[-1] > self.admmopt.residual_distance * self.dualgaps[-1]:
                    self.rho = min(self.admmopt.max_penalty, self.rho * self.admmopt.penalty_mult)
                elif self.dualgaps[-1] > self.admmopt.residual_distance * self.primalgaps[-1]:
                    self.rho = self.rho / self.admmopt.penalty_mult


    def choose_max_rho(self):
        """
        Set `self.rho` to the maximum rho value among self and neighbors.
        """
        self.rho = max(self.rho, *[msg.rho for msg in self.messages.values()])


    def local_convergence(self):
        """
        Return the current local convergence status.
        Local convergence is reached if primal gap, dual gap, and constraint mismatch with
        all neighbors are below their respective tolerance thresholds.
        """
        return self.status[self.ID]


    def all_converged(self):
        """
        Return whether this cluster and all neighbors who have sent a variable update during
        the current iteration have converged.
        """
        return self.local_convergence() and all(self.status[k] for k in self.updated[-1])


    def global_convergence(self):
        """
        Return whether global convergence has been reached, i.e. if all clusters have
        reached local convergence.
        """
        return all(self.status)


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


    def update_lamda(self):
        """
        Calculate the new Lagrangian multipliers.
        """
        self.lamda = self.lamda + self.rho * (self.flows_all - self.flow_global)


    def update_flow_global(self):
        """
        Update `self.flow_global` for all neighbors from which a msg was received, then
        update the dual gap.
        """
        self.log('Updating flow global')
        flow_global_old = deepcopy(self.flow_global)
        for k in self.updated[-1]:
            msg = self.messages[k]
            lamda = msg.lamda
            flow = msg.flow
            rho = msg.rho

            # TODO: can the indexing be improved?
            self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)] = (
                (self.lamda.loc[self.lamda.index.isin(self.flows_with_neighbor[k].index)] +
                 lamda + self.flows_with_neighbor[k] * self.rho + flow * rho +
                 self.admmopt.async_correction * self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)]) /
                (self.rho + rho + self.admmopt.async_correction))

        self.update_dualgap(flow_global_old)
        self.update_mismatch((k for k in self.neighbors if k in self.messages))


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


class AdmmMessage(object):
    def __init__(
        self,
        sender,
        flow,
        lamda,
        rho,
        flow_global,
        ):

        self.sender = sender
        self.flow = flow
        self.lamda = lamda
        self.rho = rho
        self.flow_global = flow_global


class AdmmStatusMessage:
    def __init__(self, sender, converged, terminated):
        self.sender = sender
        self.converged = converged
        self.terminated = terminated
