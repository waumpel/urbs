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
    `converged`: Flag indicating local convergence.
    `dual_tolerance`: Tolerance threshold for dual gap, scaled by number of constraints.
    `dualgaps`: List holding the dual gaps after each iteration.
    `flow_global`: `pd.Series` holding the global flow values. Index is
        `['t', 'stf', 'sit', 'sit_']`.
    `flows_all`: `pd.Series` holding the values of the local flow variables after each
        solver iteration. Index is `['t', 'stf', 'sit', 'sit_']`. Initial value is `None`.
    `flows_with_neighbor`: `pd.Series` holding the values of the local flow variables
        with each neighbor after each solver iteration. Index is
        `['t', 'stf', 'sit', 'sit_']`.Initial value is `None`.
    `ID`: ID of this subproblem (zero-based integer).
    `iteration_table`: List holding the iteration counters of all clusters as received
        through messages. If cluster `k` has reached local convergence, the `k`th entry is a
        list holding the counters of all clusters that cluster `k` has received. If it has
        not reached local convergence, the `k`th entry is just the counter of cluster `k`
        (an integer). The entry at position `self.ID` is always a list.
    `lamda`: `pd.Series` holding the Lagrange multipliers. Index is
        `['t', 'stf', 'sit', 'sit_']`.
    `logfile`: Logfile for this cluster.
    `messages`: Dict mapping each neighbor ID to the most recent message from that neighbor.
    `model`: `pyomo.ConcreteModel`.
    `n_clusters`: Total number of clusters.
    `n_neighbors`: Number of neighbors.
    `n_wait`: Number of updated neighbors required for the next iteration.
    `neighbors`: List of neighbor IDs.
    `nu`: Current iteration.
    `objective_values`: List of objective function values after each iteration.
    `primal_tolerance`: Tolerance threshold for primal gap, scaled by number of constraints.
    `primalgaps`: List of primal gaps after each iteration.
    `received`: List of sets holding the IDs of updated neighbors at the start of each
        iteration.
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
    `terminated`: Flag indicating whether the solver for this model has terminated.
    `update_flag`: Flag indicating whether the iteration table has been changed since the
        last time a msg was sent.
    """

    def __init__(
        self,
        admmopt,
        flow_global,
        ID,
        initial_values,
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
        self.converged = False
        self.dual_tolerance = admmopt.dual_tolerance * min(1, len(flow_global))
        self.dualgaps = [0]
        self.flow_global = flow_global
        self.flows_all = None
        self.flows_with_neighbor = None
        self.ID = ID
        self.iteration_table = [-1] * n_clusters
        self.iteration_table[self.ID] = [-1] * n_clusters
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
        self.received = [set()]
        self.receiving_queue = receiving_queue
        self.regions = regions
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
        self.update_flag = False


    def __del__(self):
        self.logfile.close()


    def log(self, s):
        self.logfile.write(s + '\n')
        self.logfile.flush()


    def solve_problem(self):
        self.nu += 1
        self.log('---------------')
        self.log(f'Iteration {self.nu}')
        self.log('---------------')
        self.log(f'Starting with {len(self.received[-1])} updated neighbors: {self.received[-1]}')
        self.iteration_table[self.ID][self.ID] = self.nu
        self.received.append(set())

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
        Send an `AdmmMessage` with the current status to each neighbor.
        """
        if self.terminated:
            status = 'terminated'
            self.log(f"Sending msg with status 'terminated'")
        elif self.converged:
            status = self.iteration_table
            self.log(f'Sending msg with iteration table')
        else:
            status = self.nu
            self.log(f'Sending msg with iteration counter')

        for k, que in self.sending_queues.items():

            msg = AdmmMessage(
                sender = self.ID,
                flow = self.flows_with_neighbor[k],
                lamda = self.lamda[self.lamda.index.isin(self.flows_with_neighbor[k].index)],
                rho = self.rho,
                flow_global = self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)],
                status = status,
            )
            que.put(msg)

        # Reset `update_flag` as all previous updates have now been sent.
        self.update_flag = False


    def receive(self):
        """
        Check for new messages from neighbors and update the iteration table if necessary.
        Set the `self.terminated` flag, if a termination message was received.
        Return a dictionary of newly received messages with the neighbor IDs as keys.
        """
        self.log('Checking for new messages...')
        self.update_flag = False
        new_msgs = {}

        # read accumulated messages from active neighbors
        while not self.receiving_queue.empty():
            msg = self.receiving_queue.get(block=False)
            new_msgs[msg.sender] = msg

        for msg in new_msgs.values():
            if msg.is_terminated():
                self.log('Received msg from a terminated process')
                self.terminated = True

        # store new messages
        self.messages.update(new_msgs)

        if new_msgs:
            self.log(f'New messages from {len(new_msgs)} neighbors: {list(new_msgs.keys())}')
            self.merge_tables({k: msg for k, msg in new_msgs.items() if not msg.is_terminated()})

        self.received[-1] = self.received[-1].union(new_msgs.keys())
        return new_msgs


    def merge_tables(self, msgs):
        """
        Merge the iteration tables/counters of the received `msgs` with
        `self.iteration_table`. Set the `self.update_flag` if any updates were carried out.
        """
        self.log('Merging iteration tables...')
        for sender, msg in msgs.items():
            # In case of local convergence, `msg.status` is a list.
            if msg.is_converged():
                for k in range(self.n_clusters):
                    m = self.iteration_table[k]
                    n = msg.status[k]
                    if isinstance(m, list) and isinstance(n, list):
                        for l in range(self.n_clusters):
                            if n[l] > m[l]:
                                m[l] = n[l]
                                self.update_flag = True
                    else:
                        nu = m[k] if isinstance(m, list) else m
                        nu_other = n[k] if isinstance(n, list) else n
                        if nu_other > nu:
                            self.iteration_table[k] = n
                            self.iteration_table[self.ID][k] = nu_other
                            self.update_flag = True

            # Otherwise, `msg.status` is the iteration counter of the sender, so it is
            # always more recent than the locally stored value.
            else:
                self.iteration_table[sender] = msg.status
                self.iteration_table[self.ID][sender] = msg.status
                self.update_flag = True

        if self.update_flag:
            self.log('Table was updated')

        return self.update_flag


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
        for k in self.received[-1]:
            msg = self.messages[k]
            lamda = msg.lamda
            flow = msg.flow
            rho = msg.rho

            # TODO: alpha
            # TODO: can the indexing be improved?
            self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)] = (
                (self.lamda.loc[self.lamda.index.isin(self.flows_with_neighbor[k].index)] +
                 lamda + self.flows_with_neighbor[k] * self.rho + flow * rho +
                 self.admmopt.async_correction * self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)]) /
                (self.rho + rho + self.admmopt.async_correction))

        self.dualgaps.append(self.rho * np.square(self.flow_global - flow_global_old).sum(axis=0))


    def calc_primalgap(self):
        """
        Calculate the new primal gap, append it to `self.primalgaps` and check local
        convergence and store the status in `self.converged`.
        """
        primalgap = np.square(self.flows_all - self.flow_global).sum(axis=0)
        self.log(f'Primal gap: {primalgap}')
        self.primalgaps.append(primalgap)
        self.converged = self.is_converged()


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


    def is_converged(self):
        """
        Check whether primal gap, dual gap and constraint mismatch are below their
        respective tolerance thresholds.
        """
        # primal gap
        self.log('Checking convergence...')
        if self.primalgaps[-1] > self.primal_tolerance:
            self.log('No primal convergence:')
            self.log(f'{self.primalgaps[-1]} > {self.primal_tolerance}')
            return False

        # dual gap
        if self.dualgaps[-1] > self.dual_tolerance:
            self.log('No dual convergence:')
            self.log(f'{self.dualgaps[-1]} > {self.dual_tolerance}')
            return False

        # constraint mismatch
        # TODO: update this as msgs are received, keep track of violators
        if len(self.messages) < self.n_neighbors:
            self.log('No mismatch convergence:')
            self.log('Not all neighbors have sent a msg')
            return False

        for k, msg in self.messages.items():
            mismatch_gap = np.square(
                self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)]
                - msg.flow_global
            ).sum(axis=0)
            if mismatch_gap > self.mismatch_tolerance:
                self.log(f'No mismatch convergence with neighbor {k}:')
                self.log(f'{mismatch_gap} > {self.mismatch_tolerance}')
                return False

        self.log('Local convergence!')
        return True


    def global_convergence(self):
        """
        Check global convergence, i.e. local convergence + congruence. Congruence is
        achieved when all entries of `self.iteration_table` are identical lists.
        """
        if not self.converged:
            return False
        if not all(isinstance(l, list) for l in self.iteration_table):
            return False

        self.log('\n'.join(' '.join(str(x) for x in l) for l in self.iteration_table))

        congruence = all(
            l == self.iteration_table[0]
            for l in self.iteration_table[1:]
        )
        self.log(f'Congruence: {congruence}')
        return congruence


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
        status
        ):

        self.sender = sender
        self.flow = flow
        self.lamda = lamda
        self.rho = rho
        self.flow_global = flow_global
        self.status = status


    def is_terminated(self):
        return self.status == 'terminated'


    def is_converged(self):
        return isinstance(self.status, list)
