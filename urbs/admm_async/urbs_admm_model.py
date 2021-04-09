############################################################################
# This file builds the opf_admm_model class that represents a subproblem
# ADMM algorithm parameters should be defined in AdmmOption
# Package Pypower 5.1.3 is used in this application
############################################################################

from copy import deepcopy
import queue

import numpy as np
from numpy import maximum
import pandas as pd
import pyomo.environ as pyomo


class UrbsAdmmModel(object):
    """This class encapsulates the local urbs subproblem and implements admm steps
    including x-update(solving subproblem), send data to neighbors, receive data
    from neighbors, z-update (global flows) and y-update (lambdas)
    """

    def __init__(self):
        # initialize all the fields
        self.shared_lines = None
        self.flows_all = None
        self.flows_with_neighbor = None
        self.flow_global = None
        self.sub_pyomo = None
        self.sub_persistent = None
        self.neighbors = None
        self.nneighbors = None
        self.nwait = None
        self.var = {'flow_global': None, 'rho': None}
        self.ID = None
        self.nbor = {}
        self.pipes = None
        self.sending_queues = None
        self.receiving_queues = None
        self.admmopt = AdmmOption()
        self.recvmsg = {}
        self.primalgap = [9999]
        self.dualgap = [9999]
        self.gapAll = None
        self.rho = None
        self.lamda = None
        self.received_neighbors = []


    def solve_problem(self):
        self.sub_persistent.solve(save_results=False, load_solutions=False, warmstart=True)


    # TODO: series check??
    def fix_flow_global(self):
        """
        Fix the `flow_global` variables in the solver to the values in `self.flow_global`.
        """
        for key in self.flow_global.index:
            self.sub_pyomo.flow_global[key].fix(self.flow_global.loc[key, 0])
            self.sub_persistent.update_var(self.sub_pyomo.flow_global[key])


    # TODO: series check??
    def fix_lamda(self):
        """
        Fix the `lamda` variables in the solver to the values in `self.lamda`.
        """
        for key in self.lamda.index:
            if not isinstance(self.lamda.loc[key], pd.core.series.Series):
                self.sub_pyomo.lamda[key].fix(self.lamda.loc[key])
                self.sub_persistent.update_var(self.sub_pyomo.lamda[key])
            else:
                self.sub_pyomo.lamda[key].fix(self.lamda.loc[key, 0])
                self.sub_persistent.update_var(self.sub_pyomo.lamda[key])


    def set_quad_cost(self, rho_old):
        """
        Update the objective function to reflect the difference between `self.rho` and
        `rho_old`.

        Call this method *after* `fix_flow_global`.
        """
        quadratic_penalty_change = 0
        # Hard coded transmission name: 'hvac', commodity 'Elec' for performance.
        # Caution, as these need to be adjusted if the transmission of other commodities exists!
        for key in self.flow_global.index:
            if (key[2] == 'Carbon_site') or (key[3] == 'Carbon_site'):
                quadratic_penalty_change += 0.5 * (self.rho - rho_old) * \
                    (self.sub_pyomo.e_tra_in[key, 'CO2_line', 'Carbon'] -
                     self.sub_pyomo.flow_global[key]) ** 2
            else:
                quadratic_penalty_change += 0.5 * (self.rho - rho_old) * \
                    (self.sub_pyomo.e_tra_in[key, 'hvac', 'Elec'] -
                     self.sub_pyomo.flow_global[key]) ** 2

        old_expression = self.sub_pyomo.objective_function.expr
        self.sub_pyomo.del_component('objective_function')
        self.sub_pyomo.add_component(
            'objective_function',
            pyomo.Objective(
                expr = old_expression + quadratic_penalty_change, sense=pyomo.minimize
            )
        )
        self.sub_persistent.set_objective(self.sub_pyomo.objective_function)


    def send(self):
        for k, que in self.sending_queues.items():
            # prepare the message to be sent to neighbor k
            msg = AdmmMessage()
            msg.config(self.ID, k, self.flows_with_neighbor[k], self.rho,
                       self.lamda[self.lamda.index.isin(self.flows_with_neighbor[k].index)],
                       self.gapAll)
            que.put(msg)


    def recv(self, pollrounds=5):
        self.recvmsg = {}
        twait = self.admmopt.pollWaitingtime
        recv_flag = [0] * self.nneighbors

        for _ in range(pollrounds):
            # read accumulated messages from all neighbors
            for i, (k, que) in zip(range(self.nneighbors), self.receiving_queues.items()):
                while not que.empty():
                    self.recvmsg[k] = que.get(block=False) # don't wait
                    recv_flag[i] = 1

            # break if enough neighbors have been received
            if sum(recv_flag) >= self.nwait:
                break

            # otherwise, wait for a message from the last neighbor
            k, que = list(self.receiving_queues.items())[-1]
            try:
                self.recvmsg[k] = que.get(timeout=twait)
                recv_flag[-1] = 1

            except queue.Empty:
                pass

        # store number of received neighbors
        self.received_neighbors.append(sum(recv_flag))


    def update_flow_global(self):
        """
        Update `self.flow_global` for all neighbors who sent messages. Calculate the new
        dual gap and append it to `self.dualgap`.
        """
        flow_global_old = deepcopy(self.flow_global)
        for k, msg in self.recvmsg.items():
            nborvar = msg.fields  # nborvar['flow'], nborvar['convergeTable']
            self.flow_global.loc[self.flow_global.index.isin(self.flows_with_neighbor[k].index)] = \
                (self.lamda.loc[self.lamda.index.isin(self.flows_with_neighbor[k].index)] +
                    nborvar['lambda'] + self.flows_with_neighbor[k] * self.rho + nborvar['flow'] * nborvar['rho']) \
                / (self.rho + nborvar['rho'])
        self.dualgap.append(self.rho * (np.sqrt(np.square(self.flow_global - flow_global_old).sum(axis=0)[0])))


    def update_lamda(self):
        self.lamda = self.lamda + self.rho * (self.flows_all.loc[:, [0]] - self.flow_global)


    def update_rho(self, nu):
        """
        Calculate the new primal gap, append it to `self.primalgap` and store it in
        `self.gapAll`.
        Update `self.rho` according to the current primal and dual gaps unless the
        current iteration is above `self.admmopt.rho_update_nu`.

        ### Arguments
        * `nu`: The current iteration.
        """
        self.primalgap.append(np.sqrt(np.square(self.flows_all - self.flow_global).sum(axis=0)[0]))
        # update rho (only in the first rho_iter_nu iterations)
        if nu <= self.admmopt.rho_update_nu:
            if self.primalgap[-1] > self.admmopt.mu * self.dualgap[-1]:
                self.rho = min(self.admmopt.rho_max, self.rho * self.admmopt.tau)
            elif self.dualgap[-1] > self.admmopt.mu * self.primalgap[-1]:
                self.rho = min(self.rho / self.admmopt.tau, self.admmopt.rho_max)
        # update local converge table
        self.gapAll[self.ID] = self.primalgap[-1]


    def choose_max_rho(self):
        """
        Set `self.rho` to the maximum rho value among self and neighbors.
        """
        self.rho = max(self.rho, *[msg.fields['rho'] for msg in self.recvmsg.values()])


    def is_converged(self):
        # first update local convergence table using received convergence tables
        for msg in self.recvmsg.values():
            table = msg.fields['convergeTable']
            self.gapAll = list(map(min, zip(self.gapAll, table))) # TODO: is min adequate? Should we get the most recent value instead?
        # check if all local primal gaps < tolerance
        if max(self.gapAll) < self.convergetol:
            return True
        else:
            return False


    def retrieve_boundary_flows(self):
        """
        Retrieve optimized flow values for shared lines from the solver and store them in
        `self.flows_all` and `self.flows_with_neighbor`.
        """
        e_tra_in_per_neighbor = {}

        self.sub_persistent.load_vars(self.sub_pyomo.e_tra_in[:, :, :, :, :, :])
        boundary_lines_pairs = self.shared_lines.reset_index().set_index(['Site In', 'Site Out']).index
        e_tra_in_dict = {(tm, stf, sit_in, sit_out): v.value for (tm, stf, sit_in, sit_out, tra, com), v in
                         self.sub_pyomo.e_tra_in.items() if ((sit_in, sit_out) in boundary_lines_pairs)}

        e_tra_in_dict = pd.DataFrame(list(e_tra_in_dict.values()),
                                     index=pd.MultiIndex.from_tuples(e_tra_in_dict.keys())).rename_axis(
            ['t', 'stf', 'sit', 'sit_'])

        for (tm, stf, sit_in, sit_out) in e_tra_in_dict.index:
            e_tra_in_dict.loc[(tm, stf, sit_in, sit_out), 'neighbor_cluster'] = self.shared_lines.reset_index(). \
                set_index(['support_timeframe', 'Site In', 'Site Out']).loc[(stf, sit_in, sit_out), 'neighbor_cluster']

        for neighbor in self.neighbors:
            e_tra_in_per_neighbor[neighbor] = e_tra_in_dict.loc[e_tra_in_dict['neighbor_cluster'] == neighbor]
            e_tra_in_per_neighbor[neighbor].reset_index().set_index(['t', 'stf', 'sit', 'sit_'], inplace=True)
            e_tra_in_per_neighbor[neighbor].drop('neighbor_cluster', axis=1, inplace=True)

        self.flows_all = e_tra_in_dict
        self.flows_with_neighbor = e_tra_in_per_neighbor


# ##--------ADMM parameters specification -------------------------------------
class AdmmOption(object):
    """ This class defines all the parameters to use in admm """

    def __init__(self):
        self.rho_max = 10  # upper bound for penalty rho
        self.tau_max = 1.5  # parameter for residual balancing of rho
        self.tau = 1.05  # multiplier for increasing rho
        self.zeta = 1  # parameter for residual balancing of rho
        self.theta = 0.99  # multiplier for determining whether to update rho
        self.mu = 10  # multiplier for determining whether to update rho
        self.pollWaitingtime = 0.001  # waiting time of receiving from one pipe
        self.nwaitPercent = 0.2  # waiting percentage of neighbors (0, 1]
        self.iterMaxlocal = 20  # local maximum iteration
        #self.convergetol = 365 * 10 ** 1#  convergence criteria for maximum primal gap
        self.rho_update_nu = 50 # rho is updated only for the first 50 iterations
        self.conv_rel = 0.1 # the relative convergece tolerance, to be multiplied with len(s.flow_global)


class AdmmMessage(object):
    """ This class defines the message region i sends to/receives from j """

    def __init__(self):
        self.fID = 0  # source region ID
        self.tID = 0  # destination region ID
        self.fields = {
            'flow': None,
            'rho': None,
            'lambda': None,
            'convergeTable': None}


    def config(self, f, t, var_flow, var_rho, var_lambda, gapall):  # AVall and var are local variables of f region
        self.fID = f
        self.tID = t

        self.fields['flow'] = var_flow
        self.fields['rho'] = var_rho
        self.fields['lambda'] = var_lambda
        self.fields['convergeTable'] = gapall
