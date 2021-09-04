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

    # TODO: are all attributes needed?
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

        self.ID = ID
        self.logfile = join(result_dir, f'solver-{ID}.log')
        self.admmopt = admmopt
        self.flow_global = flow_global
        self.flows_all = None
        self.flows_with_neighbor = None
        self.lamda = lamda
        self.neighbors = neighbors
        self.nu = -1
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
        solver.solve(model, tee=True, report_timing=False)
        solver_stop = time()

        objective = pyomo.value(model.objective_function)
        self._retrieve_boundary_flows(model.e_tra_in)

        return objective, solver_start, solver_stop


    def update_lamda(self) -> None:
        """
        Calculate the new Lagrangian multipliers.
        """
        self.lamda = self.lamda + self.rho * (self.flows_all - self.flow_global)


    # synchronous ADMM version
    def update_flow_global(self, updates: Dict) -> None:
        """
        Update `self.flow_global` for all neighbor => msg pairs in `updates`.
        """
        flow_global_old = deepcopy(self.flow_global)
        for k, msg in updates.items():
            self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)] = \
                (self.flows_with_neighbor[k] + msg.flow) / 2


    @staticmethod
    def calc_rho(
        nu,
        admmopt,
        rho,
        primalgap,
        primalgap_old,
        dualgap,
        ) -> float:
        """
        Update the penalty parameter according to the strategy stored in `self.admmopt`.
        """
        if admmopt.penalty_mode == 'increasing':
            if (nu > 0 and
                primalgap > admmopt.primal_decrease * primalgap_old):
                rho = min(
                    admmopt.max_penalty, rho * admmopt.penalty_mult
                )

        elif admmopt.penalty_mode == 'residual_balancing':
            if nu > 0:
                if primalgap > admmopt.residual_distance * dualgap:
                    rho = min(
                        admmopt.max_penalty, rho * admmopt.penalty_mult
                    )
                elif dualgap > admmopt.residual_distance * primalgap:
                    rho = rho / admmopt.penalty_mult

        elif admmopt.penalty_mode == 'adaptive_multiplier':
            if nu > 0:
                if primalgap > admmopt.mult_adapt * \
                   admmopt.residual_distance * dualgap:
                    mult = AdmmModel._calc_multiplier(admmopt, primalgap, dualgap)
                    rho = min(admmopt.max_penalty, rho * mult)
                elif dualgap > admmopt.residual_distance * primalgap \
                     / admmopt.mult_adapt:
                    mult = AdmmModel._calc_multiplier(admmopt, primalgap, dualgap)
                    rho = rho / mult

        return rho


    def mismatch(self, k: int, flow_global: pd.Series) -> float:
        flow_global_with_k = self.flow_global.loc[
            self.flow_global.index.isin(
                self.flows_with_neighbor[k].index
            )
        ]
        raw_mismatch_gap = norm(flow_global_with_k - flow_global)

        if self.admmopt.tolerance_mode == 'absolute':
            mismatch_gap = raw_mismatch_gap / min(1, len(self.model.flow_global))

        elif self.admmopt.tolerance_mode == 'relative':
            normalizer = max(norm(flow_global_with_k), norm(flow_global))
            if normalizer == 0:
                normalizer = 1
            mismatch_gap = norm(flow_global_with_k - flow_global) / normalizer

        return mismatch_gap


    @staticmethod
    def _calc_multiplier(admmopt, primalgap, dualgap) -> float:
        """
        Calculate the multiplier for the penalty update in `adaptive_multiplier` mode.
        """
        dualgap = dualgap
        if dualgap == 0:
            dualgap = 1

        ratio = sqrt(primalgap / (dualgap * admmopt.mult_adapt))

        if 1 <= ratio and ratio < admmopt.max_mult:
            return ratio
        elif 1 / admmopt.max_mult < ratio and ratio < 1:
            return 1 / ratio
        else:
            return admmopt.max_mult


    def _update_cost_rule(self, model):
        if model.obj.value == 'cost':
            if hasattr(model, 'objective_function'):
                model.del_component(model.objective_function)
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
        # print(f'model {self.ID}')
        # print(f'neighbors: {self.neighbors}')
        index = self.shared_lines_index

        flows_all = {}
        flows_with_neighbor = {k: {} for k in self.neighbors}

        for (tm, stf, sit_in, sit_out, tra, com), v in e_tra_in.items():
            # print(f'{sit_in}-{sit_out}')
            if (sit_in, sit_out) in zip(index['Site In'], index['Site Out']):
                k = self.shared_lines.loc[(stf, sit_in, sit_out, tra, com), 'neighbor_cluster']
                # print(f'shared with {k}')
                flows_all[(tm, stf, sit_in, sit_out)] = v.value
                flows_with_neighbor[k][(tm, stf, sit_in, sit_out)] = v.value

        flows_all = pd.Series(flows_all)
        flows_all.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)

        for k in flows_with_neighbor:
            # print(f'flows_with_neighbor[{k}]')
            # print(flows_with_neighbor[k])
            flows = pd.Series(flows_with_neighbor[k])
            flows.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)
            flows_with_neighbor[k] = flows

        self.flows_all = flows_all
        self.flows_with_neighbor = flows_with_neighbor
