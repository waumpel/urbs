############################################################################
# This file builds the opf_admm_model class that represents a subproblem
# ADMM algorithm parameters should be defined in AdmmOption
# Package Pypower 5.1.3 is used in this application
############################################################################

from copy import deepcopy
from math import sqrt
import time

import numpy as np
from numpy.linalg import norm
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

    TODO: update
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
    `sending_queues`: Dict mapping each cluster ID (except `self.ID`) to a `mp.Queue` for
        sending messages to that cluster.
    `shared_lines`: `pd.DataFrame` of inter-cluster transmission lines. A copy of a slice of
        the 'Transmision' DataFrame, enriched with the columns `cluster_from`, `cluster_to`
        and `neighbor_cluster`. Index is
        `['support_timeframe', 'Site In', 'Site Out', 'Transmission', 'Commodity']`.
    `shared_lines_index`: `shared_lines.index.to_frame()`.
    `solver`: `GurobiPersistent` solver interface to `model`.
    `status`: List holding the currently known `AdmmStatus` of a cluster, as received
        through messages.
    `status_update`: Flag indicating whether `self.status` has been updated since
        the last time a status message has been sent.
    `updated`: List of sets holding the neighbors who have sent new variables before
        the start of each iteration (starts with an empty set as the single entry).
    """

    def __init__(
        self,
        admmopt,
        flow_global,
        lamda,
        model,
        neighbors,
        shared_lines,
        shared_lines_index,
        threads=None,
    ):
        self.admmopt = admmopt
        self.dualgap = np.nan
        self.flow_global = flow_global
        self.flows_all = None
        self.flows_with_neighbor = None
        self.lamda = lamda
        self.model = model
        self.neighbors = neighbors
        self.nu = -1
        self.primalgap = None
        self.primalgap_old = None
        self.rho = admmopt.rho
        self.shared_lines = shared_lines
        self.shared_lines_index = shared_lines_index

        if threads is None:
            threads = 1
        self.solver = SolverFactory('gurobi_persistent')
        self.solver.set_instance(model, symbolic_solver_labels=False)
        self.solver.set_options("NumericFocus=3")
        self.solver.set_options("Crossover=0")
        self.solver.set_options("Method=2")
        self.solver.set_options(f"Threads={threads}")


    def solve_iteration(self):
        """
        Start a new iteration and solve the optimization problem.
        """
        self.nu += 1

        solver_start = time.time()
        self.solver.solve(save_results=False, load_solutions=False, warmstart=True)
        solver_stop = time.time()

        objective = self.solver._solver_model.objval
        self._retrieve_boundary_flows()
        self._update_primalgap()

        return objective, self.primalgap, self.dualgap, self.rho, solver_start, solver_stop


    def update_lamda(self):
        """
        Calculate the new Lagrangian multipliers.
        """
        self.lamda = self.lamda + self.rho * (self.flows_all - self.flow_global)


    def update_flow_global(self, updates):
        """
        Update `self.flow_global` for all neighbors from which a msg was received, then
        update the dual gap.
        """
        flow_global_old = deepcopy(self.flow_global)
        for k, msg in updates.items():
            lamda = msg.lamda
            flow = msg.flow
            rho = msg.rho

            # TODO: can the indexing be improved?
            self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)] = (
                (self.lamda.loc[self.lamda.index.isin(self.flows_with_neighbor[k].index)] +
                 lamda + self.flows_with_neighbor[k] * self.rho + flow * rho +
                 self.admmopt.async_correction * self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)]) /
                (self.rho + rho + self.admmopt.async_correction))

        self._update_dualgap(flow_global_old)


    def update_rho(self, neighbor_rhos):
        """
        Increase `self.rho` if the primal gap does not decrease sufficiently.
        """
        if self.admmopt.penalty_mode == 'increasing':
            if (self.nu > 0 and
                self.primalgap > self.admmopt.primal_decrease * self.primalgap_old):
                self.rho = min(
                    self.admmopt.max_penalty, self.rho * self.admmopt.penalty_mult
                )

            # choose max among neighbors
            self.rho = max(self.rho, *neighbor_rhos)

        elif self.admmopt.penalty_mode == 'residual_balancing':
            if self.nu > 0:
                if self.primalgap > self.admmopt.residual_distance * self.dualgap:
                    self.rho = min(
                        self.admmopt.max_penalty, self.rho * self.admmopt.penalty_mult
                    )
                elif self.dualgap > self.admmopt.residual_distance * self.primalgap:
                    self.rho = self.rho / self.admmopt.penalty_mult

        elif self.admmopt.penalty_mode == 'adaptive_multiplier':
            if self.nu > 0:
                if self.primalgap > self.admmopt.mult_adapt * \
                   self.admmopt.residual_distance * self.dualgap:
                    mult = self._calc_multiplier()
                    self.rho = min(self.admmopt.max_penalty, self.rho * mult)
                elif self.dualgap > self.admmopt.residual_distance * self.primalgap \
                     / self.admmopt.mult_adapt:
                    mult = self._calc_multiplier()
                    self.rho = self.rho / mult


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


    def _retrieve_boundary_flows(self):
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


    def _update_primalgap(self):
        self.primalgap_old = self.primalgap

        raw_primalgap = norm(self.flows_all - self.flow_global)
        if self.admmopt.tolerance_mode == 'absolute':
            primalgap = raw_primalgap / min(1, len(self.flow_global))
        elif self.admmopt.tolerance_mode == 'relative':
            normalizer = max(
                norm(self.flows_all),
                norm(self.flow_global)
            )
            if normalizer == 0:
                normalizer = 1
            primalgap = raw_primalgap / normalizer

        self.primalgap = primalgap


    def _update_dualgap(self, flow_global_old):
        """
        Calculate the new dual gap.
        """
        raw_dualgap = self.rho * norm(self.flow_global - flow_global_old)

        if self.admmopt.tolerance_mode == 'absolute':
            dualgap = raw_dualgap / min(1, len(self.flow_global))
        elif self.admmopt.tolerance_mode == 'relative':
            normalizer = norm(self.lamda)
            if normalizer == 0:
                normalizer = 1
            dualgap = raw_dualgap / normalizer
        self.dualgap = dualgap


    def _calc_multiplier(self):
        dualgap = self.dualgap
        if dualgap == 0:
            dualgap = 1

        ratio = sqrt(self.primalgap / (self.dualgap * self.admmopt.mult_adapt))

        if 1 <= ratio and ratio < self.admmopt.max_mult:
            return ratio
        elif 1 / self.admmopt.max_mult < ratio and ratio < 1:
            return 1 / ratio
        else:
            return self.admmopt.max_mult
