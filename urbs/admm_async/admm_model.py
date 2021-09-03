############################################################################
# This file builds the opf_admm_model class that represents a subproblem
# ADMM algorithm parameters should be defined in AdmmOption
# Package Pypower 5.1.3 is used in this application
############################################################################

from copy import deepcopy
from math import sqrt
from os.path import join
from time import time
from typing import Dict, List, Tuple
from urbs.model import cost_rule_sub

import numpy as np
from numpy.linalg import norm
import pandas as pd
import pyomo.environ as pyomo

from .admm_option import AdmmOption


# TODO: docstrings
class AdmmModel:
    """
    Encapsulates an urbs subproblem and implements ADMM steps.

    Attributes
        - `admmopt`: `AdmmOption` object
        - `dualgap`: Current dual gap
        - `flow_global`: `pd.Series` containing the global flow values. Index is
          `['t', 'stf', 'sit', 'sit_']`.
        - `flows_all`: `pd.Series` containing the current values of the local flow variables.
          Index is `['t', 'stf', 'sit', 'sit_']`.
        - `flows_with_neighbor`: Dict containing containing the current values of the local
          flow variables with each neighbor. Each entry is a `pd.Series` with index
          ['t', 'stf', 'sit', 'sit_']`.
        - `lamda`: `pd.Series` containing the Lagrange multipliers. Index is
          `['t', 'stf', 'sit', 'sit_']`.
        - `model`: `pyomo.ConcreteModel`
        - `neighbors`: List of neighbor IDs
        - `nu`: Current iteration
        - `primalgap`: Current primal gap
        - `primalgap_old`: Last primal gap
        - `rho`: Quadratic penalty coefficient
        - `shared_lines`: `pd.DataFrame` of inter-cluster transmission lines. A copy of a
            slice of the 'Transmision' DataFrame, enlarged with the columns `cluster_from`,
            `cluster_to` and `neighbor_cluster`.
        - `shared_lines_index`: Index of `shared_lines` as a `DataFrame`
    """

    def __init__(
        self,
        ID,
        result_dir,
        admmopt: AdmmOption,
        neighbors: List[int],
        shared_lines: pd.DataFrame,
        shared_lines_index: pd.DataFrame,
        flow_global: pd.Series,
        lamda: pd.Series,
        ) -> None:

        self.logfile = join(result_dir, f'solver-{ID}.log')

        self.admmopt = admmopt
        self.dualgap = np.nan
        self.flow_global = flow_global
        self.flows_all = None
        self.flows_with_neighbor = None
        self.lamda = lamda
        self.neighbors = neighbors
        self.nu = -1
        self.primalgap = None
        self.primalgap_old = None
        self.rho = admmopt.rho
        self.shared_lines = shared_lines
        self.shared_lines_index = shared_lines_index


    def solve_iteration(self, solver, model) -> Tuple:
        """
        Start a new iteration and solve the optimization problem.

        Return the objective value, primal gap, dual gap, penalty parameter (rho),
        start time and stop time.
        """
        self.nu += 1
        self._update_cost_rule(model)
        solver.set_options(f"LogFile={self.logfile}")

        solver_start = time()
        solver.solve(model, save_results=False, load_solutions=False, warmstart=True,
                     tee=True, report_timing=False)
        solver_stop = time()

        objective = pyomo.value(model.obj)
        self._retrieve_boundary_flows(model.e_tra_in)
        self._update_primalgap()

        return objective, self.primalgap, self.dualgap, self.rho, solver_start, solver_stop


    def update_lamda(self) -> None:
        """
        Calculate the new Lagrangian multipliers.
        """
        self.lamda = self.lamda + self.rho * (self.flows_all - self.flow_global)


    def update_flow_global(self, updates: Dict) -> None:
        """
        Update `self.flow_global` for all neighbor => msg pairs in `updates`, then
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


    def update_rho(self, neighbor_rhos: List[float]) -> None:
        """
        Update the penalty parameter according to the strategy stored in `self.admmopt`.
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


    def _update_primalgap(self) -> None:
        """
        Calculate the new primal gap.
        """
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


    def _update_dualgap(self, flow_global_old: pd.Series) -> None:
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


    def _calc_multiplier(self) -> float:
        """
        Calculate the multiplier for the penalty update in `adaptive_multiplier` mode.
        """
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


    def _update_cost_rule(self, model):
        if model.obj.value == 'cost':
            model.objective_function = pyomo.Objective(
                rule=cost_rule_sub(self.flow_global, self.lamda, self.rho),
                sense=pyomo.minimize,
                doc='minimize(cost = sum of all cost types) + penalty'
            )
        else:
            raise NotImplementedError("Objectives other than 'cost' are not supported.")


    def _retrieve_boundary_flows(
        self,
        e_tra_in,
        ) -> None:
        """
        Retrieve optimized flow values for shared lines from the `solver` and store them in
        `model.flows_all` and `model.flows_with_neighbor`.

        Arguments:
            - `model: AdmmModel
            - `solver`: pyomo solver
        """
        index = self.shared_lines_index

        flows_all = {}
        flows_with_neighbor = {k: {} for k in self.neighbors}

        for (tm, stf, sit_in, sit_out, tra, com), v in e_tra_in.items():
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
