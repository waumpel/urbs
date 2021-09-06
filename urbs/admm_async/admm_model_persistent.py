############################################################################
# This file builds the opf_admm_model class that represents a subproblem
# ADMM algorithm parameters should be defined in AdmmOption
# Package Pypower 5.1.3 is used in this application
############################################################################

from copy import deepcopy
from time import time
from typing import Dict, List, Tuple

import numpy as np
from numpy.linalg import norm
import pandas as pd
import pyomo.environ as pyomo
from pyomo.environ import SolverFactory

from .admm_messages import AdmmVariableMessage
from .admm_model import AdmmModel
from .admm_option import AdmmOption


class AdmmModelPersistent(AdmmModel):
    """
    Encapsulates an urbs subproblem and implements ADMM steps.

    Attributes
        - `dualgap`: Current dual gap
        - `model`: `pyomo.ConcreteModel`
        - `nu`: Current iteration
        - `primalgap`: Current primal gap
        - `primalgap_old`: Last primal gap
        - `solver`: `GurobiPersistent` solver interface to `model`

    See also: `AdmmModel`, `AdmmWorker`
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

        self.dualgap = np.nan
        self.nu = -1
        self.primalgap = None
        self.primalgap_old = None

        self.solver = SolverFactory('gurobi_persistent')
        self.solver.set_instance(model, symbolic_solver_labels=False)
        self.solver.set_options(f"LogToConsole=0")
        self.solver.set_options(f"LogFile={self.logfile}")
        self.solver.set_options("NumericFocus=3")
        self.solver.set_options("Crossover=0")
        self.solver.set_options("Method=2")
        self.solver.set_options(f"Threads={threads}")


    # override
    def solve_iteration(self) -> Tuple[float, float, float, float, float, float]:
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
    def update_flow_global(self, updates: Dict[int, AdmmVariableMessage]) -> None:
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

        After the update, if the penalty mode is `increasing`, the maximum is picked from
        the newly computed value and `neighbor_rhos`.
        """
        self.rho = AdmmModel.calc_rho(
            self.nu, self.admmopt, self.rho, self.primalgap, self.primalgap_old, self.dualgap
        )
        if self.admmopt.penalty_mode == 'increasing':
            self.rho = max(self.rho, *neighbor_rhos)


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


    # override
    def _update_cost_rule(self) -> None:
        """
        Update the objective function of `model` to reflect changes to the global flow
        values, Lagrange multipliers, and penalty parameter.
        """
        super()._update_cost_rule(self.model)
        self.solver.set_objective(self.model.objective_function)


    # override
    def _retrieve_boundary_flows(self) -> None:
        """
        Retrieve optimized flow values from the solver and store them in
        `self.flows_all` and `self.flows_with_neighbor`.
        """
        self.solver.load_vars(self.model.e_tra_in[:, :, :, :, :, :])
        super()._retrieve_boundary_flows(self.model.e_tra_in)
