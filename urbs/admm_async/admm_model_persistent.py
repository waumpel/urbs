############################################################################
# This file builds the opf_admm_model class that represents a subproblem
# ADMM algorithm parameters should be defined in AdmmOption
# Package Pypower 5.1.3 is used in this application
############################################################################

from copy import deepcopy
from time import time
from typing import Dict, List, Tuple

import pandas as pd
import pyomo.environ as pyomo
from pyomo.environ import SolverFactory

from .admm_model import AdmmModel
from .admm_option import AdmmOption


# TODO: docstrings
class AdmmModelPersistent(AdmmModel):
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
        - `solver`: `GurobiPersistent` solver interface to `model`

    See also: `AdmmWorker`
    """

    def __init__(
        self,
        ID: int,
        result_dir: str,
        admmopt: AdmmOption,
        model: pyomo.ConcreteModel,
        neighbors: List[int],
        shared_lines: pd.DataFrame,
        shared_lines_index: pd.DataFrame,
        flow_global: pd.Series,
        lamda: pd.Series,
        threads: int=1,
    ) -> None:

        super().__init__(
            ID,
            result_dir,
            admmopt,
            neighbors,
            shared_lines,
            shared_lines_index,
            flow_global,
            lamda,
        )

        self.model = model

        self.solver = SolverFactory('gurobi_persistent')
        self.solver.set_instance(model, symbolic_solver_labels=False)
        self.solver.set_options(f"LogToConsole=0")
        self.solver.set_options(f"LogFile={self.logfile}")
        self.solver.set_options("NumericFocus=3")
        self.solver.set_options("Crossover=0")
        self.solver.set_options("Method=2")
        self.solver.set_options(f"Threads={threads}")


    # override
    def solve_iteration(self) -> Tuple:
        """
        Start a new iteration and solve the optimization problem.

        Return the objective value, primal gap, dual gap, penalty parameter (rho),
        start time and stop time.
        """
        self.nu += 1
        self._update_cost_rule()

        solver_start = time()
        self.solver.solve(save_results=False, load_solutions=False, warmstart=True,
                          tee=True, report_timing=False)
        solver_stop = time()

        objective = self.solver._solver_model.objval
        self._retrieve_boundary_flows()
        self._update_primalgap()

        return objective, self.primalgap, self.dualgap, self.rho, solver_start, solver_stop


    # override
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


    # override
    def _update_cost_rule(self) -> None:
        """
        Update those components of `self.model` that use `cost_rule_sub` to reflect
        changes to `self.flow_global`, `self.lamda` and `self.rho`.
        Currently only supports models with `cost` objective, i.e. only the objective
        function is updated.
        """
        super()._update_cost_rule(self.model)
        self.solver.set_objective(self.model.objective_function)


    # override
    def _retrieve_boundary_flows(self) -> None:
        self.solver.load_vars(self.model.e_tra_in[:, :, :, :, :, :])
        super()._retrieve_boundary_flows(self.model.e_tra_in)
