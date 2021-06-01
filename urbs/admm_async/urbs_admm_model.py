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
    `dual_tolerance`: Tolerance threshold for dual gap, scaled by number of constraints.
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
    `messages`: Dict mapping each neighbor ID to the most recent message from that neighbor.
    `mismatch_convergence`: Dict mapping each neighbor ID to a flag indicating the current
        mismatch convergence status.
    `mismatch_tolerance`: Tolerance threshold for constraint mismatch, scaled by number of
        constraints.
    `model`: `pyomo.ConcreteModel`.
    `n_clusters`: Total number of clusters.
    `n_neighbors`: Number of neighbors.
    `n_wait`: Number of updated neighbors required for the next iteration.
    `neighbors`: List of neighbor IDs.
    `nu`: Current iteration.
    `objective_values`: List of objective function values after each iteration.
    `primal_tolerance`: Tolerance threshold for primal gap, scaled by number of constraints.
    `primalgaps`: List of primal gaps after each iteration.
    `receiving_queue`: `mp.Queue` for receiving messages from neighbors.
    `regions`: List of region names in this subproblem.
    `result_dir`: Result directory.
    `rho`: Quadratic penalty coefficient.
    `scenario_name`: Scenario name.
    `sending_queues`: Dict mapping each neighbor ID to a `mp.Queue` for sending messages to
        that neighbor.
    `shared_lines`: `pd.DataFrame` of inter-cluster transmission lines. A copy of a slice of
        the 'Transmision' DataFrame, enriched with the columns `cluster_from`, `cluster_to`
        and `neighbor_cluster`. Index is
        `['support_timeframe', 'Site In', 'Site Out', 'Transmission', 'Commodity']`.
    `shared_lines_index`: `shared_lines.index.to_frame()`.
    `solver`: `GurobiPersistent` solver interface to `model`.
    `status`: List of lists, each holding the currently known status of one cluster,
        as received through messages. Each list has three entries: iteration counter,
        local convergence flag, and consensus flag.
    `terminated`: Flag indicating whether the solver for this model has terminated.
    `status_update`: Flag indicating whether `status` has been modified since the
        last time a message was sent.
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
        receiving_queue,
        regions,
        result_dir,
        scenario_name,
        sending_queues,
        shared_lines,
        shared_lines_index,
    ):
        self.admmopt = admmopt
        self.dual_tolerance = admmopt.dual_tolerance * min(1, len(flow_global))
        self.dualgaps = [0]
        self.flow_global = flow_global
        self.flows_all = None
        self.flows_with_neighbor = None
        self.ID = ID
        self.lamda = lamda
        self.logfile = open(join(result_dir, f'process-{ID}.log'), 'w', encoding='utf8')
        self.messages = {}
        self.mismatch_tolerance = admmopt.mismatch_tolerance * min(1, len(flow_global))
        self.model = model
        self.n_clusters = n_clusters
        self.n_neighbors = len(neighbors)
        self.n_wait = ceil(self.n_neighbors * admmopt.wait_percent)
        self.neighbors = neighbors
        self.nu = -1
        self.objective_values = []
        self.primal_tolerance = admmopt.primal_tolerance * min(1, len(flow_global))
        self.primalgaps = []
        self.receiving_queue = receiving_queue
        self.regions = regions
        self.result_dir = result_dir
        self.rho = admmopt.rho
        self.scenario_name = scenario_name
        self.sending_queues = sending_queues
        self.shared_lines = shared_lines
        self.shared_lines_index = shared_lines_index

        self.solver = SolverFactory('gurobi_persistent')
        self.solver.set_instance(model, symbolic_solver_labels=False)
        self.solver.set_gurobi_param('Method', 2)
        self.solver.set_gurobi_param('Threads', 1)

        self.status = [[-1, False, False]] * n_clusters
        self.terminated = False
        self.status_update = False
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

        self.status[self.ID] = [self.nu, False, False]
        self.updated.append(set())
        self.mismatch_convergence = {k: False for k in self.neighbors}

        self.solver.solve(save_results=False, load_solutions=False, warmstart=True)
        self.objective_values.append(self.solver._solver_model.objval) # TODO: use public method instead

        self.log(f'Objective value: {self.objective_values[-1]}')


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


    def send(self):
        """
        Send an `AdmmMessage` with the current variables and status to each neighbor.
        """
        self.log('Sending full update')

        for k, que in self.sending_queues.items():
            msg = AdmmMessage(
                sender = self.ID,
                flow = self.flows_with_neighbor[k],
                lamda = self.lamda[self.lamda.index.isin(self.flows_with_neighbor[k].index)],
                rho = self.rho,
                flow_global = self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)],
                status = self.status,
                terminated = self.terminated
            )
            que.put(msg)

        # Reset `status_update` as all previous updates have now been sent.
        self.status_update = False


    def send_status(self):
        """
        Send an `AdmmStatusMessage` with the current status to each neighbor.
        """
        self.log('Sending status update')

        msg = AdmmStatusMessage(
            sender = self.ID,
            status = self.status,
            terminated = self.terminated
        )
        for que in self.sending_queues.values():
            que.put(msg)

        # Reset `status_update` as all previous updates have now been sent.
        self.status_update = False


    def receive(self):
        """
        Check for new messages from neighbors, merging status messages and full messages.
        Check for termination signals, update mismatch convergence and status.
        Return two sets, `full` and `status`, of neighbor IDs who have sent full messages or
        status messages.
        """
        self.log('Checking for new messages...')

        full = set()
        status = set()

        # read accumulated messages from active neighbors
        while not self.receiving_queue.empty():
            msg = self.receiving_queue.get(block=False)
            if isinstance(msg, AdmmMessage):
                self.messages[msg.sender] = msg
                full.add(msg.sender)
            elif isinstance(msg, AdmmStatusMessage):
                self.messages[msg.sender].update(msg)
                status.add(msg.sender)
            else:
                raise RuntimeError(f'Received a msg of unrecognized type: {type(msg)}')

            if msg.terminated:
                self.log('Received msg from a terminated process')
                self.terminated = True

        senders = full.union(status)
        self.updated[-1] = self.updated[-1].union(full)

        if senders:
            self.log(f'New messages from {len(senders)} neighbors: {senders}')
            self.log(f'Thereof full updates: {full}')
        else:
            self.log('No new messages.')

        if full:
            self.update_mismatch(full)

        if senders:
            self.update_status(senders)

        return full, status


    def update_status(self, senders):
        """
        Update `self.status` to reflect new information obtained from neighbors whose IDs
        are contained in `senders`.
        Update the current consensus status.

        Return whether status was modified.
        """
        self.log('Updating status...')

        for sender in senders:
            msg = self.messages[sender]

            self.log(f'Status from {sender}:')
            self.log('\n'.join(str(s) for s in msg.status))

            for k in range((self.n_clusters)):
                theirs = msg.status[k]
                ours = self.status[k]
                # compare iteration counters
                if theirs[0] > ours[0]:
                    self.status[k] = theirs
                    self.status_update = True
                # If the iteration counters are equal, then the received status may stem
                # from a more recent message of the same iteration.
                # In that case, local convergence and status can only have been rendered
                # True, not False.
                elif theirs[0] == ours[0]:
                    if theirs[1] and not ours[1]:
                        ours[1] = True
                        self.status_update = True
                    if theirs[2] and not ours[2]:
                        ours[2] = True
                        self.status_update = True

            # always update status of the sender
            if self.status[sender] != msg.status[sender]:
                self.status[sender] = msg.status[sender]
                self.status_update = True

        if self.status_update:
            self.log('New status:')
            self.log('\n'.join(str(s) for s in self.status))
            self.update_consensus()

        else:
            self.log('Nothing to update.')

        return self.status_update


    def update_lamda(self):
        """
        Calculate the new Lagrangian multipliers.
        """
        self.lamda = self.lamda + self.rho * (self.flows_all - self.flow_global)


    def update_flow_global(self):
        """
        Update `self.flow_global` for all neighbors from which a msg was received. Calculate
        the new dual gap and append it to `self.dualgaps`.
        """
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

        self.dualgaps.append(self.rho * np.square(self.flow_global - flow_global_old).sum(axis=0))
        # No need to call `update_convergence` here; this is done in the next call to
        # `update_primalgap`.


    def update_primalgap(self):
        """
        Calculate the new primal gap and append it to `self.primalgaps`.
        Update the current convergence status.
        """
        primalgap = np.square(self.flows_all - self.flow_global).sum(axis=0)
        self.log(f'Primal gap: {primalgap}')
        self.primalgaps.append(primalgap)
        self.update_convergence()


    def update_rho(self, nu):
        """
        Update `self.rho` according to the new primal and dual gaps unless the
        current iteration, `self.nu`, is above `self.admmopt.penalty_iter`.
        """
        primalgap = self.primalgaps[-1]
        dualgap = self.dualgaps[-1]

        if self.nu <= self.admmopt.penalty_iter:
            if primalgap > self.admmopt.penalty_tolerance * dualgap:
                self.rho *= self.admmopt.penalty_mult
            elif dualgap > self.admmopt.penalty_tolerance * primalgap:
                self.rho /= self.admmopt.penalty_mult


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
        self.rho = max(self.rho, *[msg.rho for msg in self.messages.values()])


    def update_mismatch(self, senders):
        """
        Update `self.mismatch_convergence` to reflect new information obtained from
        neighbors whose IDs are contained in `senders`.
        Update the current convergence status.
        """
        self.log('Updating mismatch convergence...')
        for k in senders:
            msg = self.messages[k]
            mismatch_gap = np.square(
                self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)]
                - msg.flow_global
            ).sum(axis=0)
            if mismatch_gap > self.mismatch_tolerance:
                self.log(f'No mismatch convergence with neighbor {k}:')
                self.log(f'{mismatch_gap} > {self.mismatch_tolerance}')
                self.mismatch_convergence[k] = False
            else:
                self.log(f'Reached mismatch convergence with neighbor {k}:')
                self.log(f'{mismatch_gap} <= {self.mismatch_tolerance}')
                self.mismatch_convergence[k] = True

        self.log(f'Current mismatch status: {self.mismatch_convergence}')
        self.update_convergence()


    def update_convergence(self):
        """
        Update the local convergence flag within `self.status`.
        Local convergence is reached if primal gap, dual gap, and constraint mismatch with
        all neighbors are below their respective tolerance thresholds.
        """
        # primal gap
        self.log('Checking local convergence...')
        if self.primalgaps[-1] > self.primal_tolerance:
            self.log('No primal convergence:')
            self.log(f'{self.primalgaps[-1]} > {self.primal_tolerance}')
            self.set_local_convergence(False)
            return False

        # dual gap
        if self.dualgaps[-1] > self.dual_tolerance:
            self.log('No dual convergence:')
            self.log(f'{self.dualgaps[-1]} > {self.dual_tolerance}')
            self.set_local_convergence(False)
            return False

        # constraint mismatch
        if not all(self.mismatch_convergence.values()):
            self.log('No mismatch convergence')
            self.set_local_convergence(False)
            return False

        self.log('Local convergence!')
        self.set_local_convergence(True)
        return True


    def local_convergence(self):
        """
        Return the current local convergence status.
        """
        return self.status[self.ID][1]


    def set_local_convergence(self, value):
        """
        Set the current local convergence status.
        Update the current consensus status.
        """
        old_value = self.local_convergence()
        self.status[self.ID][1] = value
        if value != old_value:
            self.status_update = True
        if value:
            self.update_consensus()


    def update_consensus(self):
        """
        Update the current consensus flag within `self.status`.
        Consensus is reached when this cluster and all of its neighbors have reached local
        convergence, and the iteration counter of this cluster received in messages from
        all neighbors is equal to `self.nu`.

        """
        self.log('Checking consensus...')
        result = self.local_convergence() and all(
            k in self.messages and
            msg.local_convergence() and
            msg.status[self.ID][0] == self.nu
            for k, msg in self.messages.items()
        )

        if result:
            self.log('Reached consensus!')
        else:
            self.log('No consensus.')

        self.set_consensus(result)


    def consensus(self):
        """
        Return the current consensus status.
        """
        return self.status[self.ID][2]


    def set_consensus(self, value):
        """
        Set the current consensus status.
        """
        old_value = self.consensus()
        self.status[self.ID][2] = value
        if value != old_value:
            self.status_update = True


    def all_converged(self):
        """
        Return whether this cluster and all neighbors who have sent a full message during
        the current iteration have converged.
        """
        return self.local_convergence() and all(
            self.messages[k].local_convergence() for k in self.updated[-1]
        )


    def global_convergence(self):
        """
        Return whether global convergence has been reached, i.e. if all clusters have
        reached consensus.
        """
        self.log('Checking global convergence...')
        result = all(s[2] for s in self.status)
        if result:
            self.log('Global convergence!')
        else:
            self.log('No global convergence.')
        return result


class AdmmOption(object):
    def __init__(self,
        async_correction,
        dual_tolerance,
        max_iter,
        mismatch_tolerance,
        penalty_iter,
        penalty_mult,
        penalty_tolerance,
        primal_tolerance,
        rho,
        wait_percent,
        wait_time,
    ):
        self.async_correction = async_correction
        self.dual_tolerance = dual_tolerance
        self.max_iter = max_iter
        self.mismatch_tolerance = mismatch_tolerance
        self.penalty_iter = penalty_iter
        self.penalty_mult = penalty_mult
        self.penalty_tolerance = penalty_tolerance
        self.primal_tolerance = primal_tolerance
        self.rho = rho
        self.wait_percent = wait_percent
        self.wait_time = wait_time


class AdmmMessage(object):
    def __init__(
        self,
        sender,
        flow,
        lamda,
        rho,
        flow_global,
        status,
        terminated,
        ):

        self.sender = sender
        self.flow = flow
        self.lamda = lamda
        self.rho = rho
        self.flow_global = flow_global
        self.status = status
        self.terminated = terminated


    def update(self, msg):
        """
        Merge the `AdmmStatusMessage msg` into this message.
        """
        if isinstance(msg, AdmmStatusMessage):
            self.status = msg.status
            self.terminated = msg.terminated
        else:
            raise TypeError(f'msg has unexpected type {type(msg)}, should be AdmmStatusMessage')


    def local_convergence(self):
        return self.status[self.sender][1]


class AdmmStatusMessage:
    def __init__(self, sender, status, terminated):
        self.sender = sender
        self.status = status
        self.terminated = terminated
