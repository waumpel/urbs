from math import ceil
from multiprocessing import Queue
from time import sleep

import numpy as np
from numpy.linalg import norm
import pandas as pd

from urbs.admm_async.admm_messages import (
    AdmmIterationResult, AdmmVariableMessage, AdmmStatusMessage, AdmmStatus
)
from .urbs_admm_model import UrbsAdmmModel
import urbs.model


class AdmmWorker:

    def __init__(self, ID: int, output, admmopt, n_clusters, neighbors, queues) -> None:
        self.ID: int = ID
        self.log_prefix: str = f'AdmmWorker[{ID}]'
        self.output: Queue = output
        self.admmopt = admmopt
        self.n_clusters = n_clusters

        self.neighbors = neighbors
        self.n_neighbors = len(neighbors)
        self.n_wait = ceil(self.n_neighbors * admmopt.wait_percent)

        self.receiving_queue = queues[ID]
        self.sending_queues = {i: queues[i] for i in range(len(queues)) if i != ID}
        self.neighbor_queues = {k: queues[k] for k in neighbors}

        self.messages = {}
        self.updated = set()
        self.status = [AdmmStatus.NO_CONVERGENCE] * n_clusters
        self.status_update = False
        self.mismatches = {k: np.nan for k in neighbors}
        self.model: UrbsAdmmModel = None


    def get_status(self):
        return self.status[self.ID]


    def _set_status(self, value):
        old_value = self.get_status()
        if old_value != AdmmStatus.TERMINATED and old_value != value:
            self.status[self.ID] = value
            self.status_update = True


    def local_convergence(self):
        """
        Return the current local convergence status.
        Local convergence is reached if primal gap, dual gap, and constraint mismatch with
        all neighbors are below their respective tolerance thresholds.
        """
        return self.get_status() in [
            AdmmStatus.LOCAL_CONVERGENCE, AdmmStatus.GLOBAL_CONVERGENCE
        ]


    def all_converged(self):
        """
        Return whether this cluster and all neighbors who have sent a variable update during
        the current iteration have converged.
        """
        return self.local_convergence() and all(
            self.status[k] in [AdmmStatus.LOCAL_CONVERGENCE, AdmmStatus.GLOBAL_CONVERGENCE]
            for k in self.updated)


    def all_global_convergence(self):
        """
        Return whether all clusters have reached global convergence.
        """
        return all(s == AdmmStatus.GLOBAL_CONVERGENCE for s in self.status)


    def max_mismatch(self):
        return np.nan if np.nan in self.mismatches.values() else \
               float(max(self.mismatches.values()))


    @staticmethod
    def run_worker(
        ID,
        output,
        data_all,
        timesteps,
        dt,
        objective,
        year,
        initial_values,
        admmopt,
        n_clusters,
        sites,
        neighbors,
        shared_lines,
        internal_lines,
        cluster_from,
        cluster_to,
        neighbor_cluster,
        queues,
        hoursPerPeriod=None,
        weighting_order=None,
        threads=None,
        ) -> None:

        print('Creating AdmmWorker object')

        worker = AdmmWorker(ID, output, admmopt, n_clusters, neighbors, queues)
        worker.run(
            data_all,
            timesteps,
            dt,
            objective,
            year,
            initial_values,
            sites,
            shared_lines,
            internal_lines,
            cluster_from,
            cluster_to,
            neighbor_cluster,
            hoursPerPeriod,
            weighting_order,
            threads,
        )


    def run(
        self,
        data_all,
        timesteps,
        dt,
        objective,
        year,
        initial_values,
        sites,
        shared_lines,
        internal_lines,
        cluster_from,
        cluster_to,
        neighbor_cluster,
        hoursPerPeriod=None,
        weighting_order=None,
        threads=None,
        ) -> None:

        self.model = self._create_model(
            data_all,
            timesteps,
            dt,
            objective,
            year,
            initial_values,
            sites,
            shared_lines,
            internal_lines,
            cluster_from,
            cluster_to,
            neighbor_cluster,
            hoursPerPeriod=hoursPerPeriod,
            weighting_order=weighting_order,
            threads=threads,
        )

        self._log(f'Starting subproblem for regions {sites}.')
        input('Continue?')

        for nu in range(self.admmopt.max_iter):
            result = self._run_iteration(nu)
            if result in [AdmmStatus.TERMINATED, AdmmStatus.GLOBAL_CONVERGENCE]:
                break

        self.output.put(AdmmStatusMessage(self.ID, self.status))


    def _run_iteration(self, nu):
        self._log(f'Iteration {nu}')

        self.updated = set()

        objective, primalgap, dualgap, penalty, start, stop = self.model.solve_iteration()
        self._log(f'Iteration {nu} solved in {(stop - start):.2f} s')
        self._update_convergence()

        self._receive()
        if self.get_status() == AdmmStatus.TERMINATED:
            return AdmmStatus.TERMINATED

        if self.status_update:
            self._send_status()

        self._send_variables()

        if self.all_global_convergence():
            self._log(f'Global convergence at iteration {nu}!')
            self.output.put(AdmmIterationResult(
                self.ID,
                nu,
                start,
                stop,
                primalgap,
                dualgap,
                self.max_mismatch(), # was updated in last call to `_receive` # TODO: rename
                objective,
                penalty,
            ))
            return AdmmStatus.GLOBAL_CONVERGENCE

        while len(self.updated) < self.n_wait or (self.all_converged()):

            sleep(self.admmopt.wait_time)
            senders = self._receive()
            if self.get_status() == AdmmStatus.TERMINATED:
                return AdmmStatus.TERMINATED
            if senders:
                if self.status_update:
                    self._send_status()
                if self.all_global_convergence():
                    self._log(f'Global convergence at iteration {nu}!')
                    self.output.put(AdmmIterationResult(
                        self.ID,
                        nu,
                        start,
                        stop,
                        primalgap,
                        dualgap,
                        self.max_mismatch(), # was updated in last call to `_receive` # TODO: rename
                        objective,
                        penalty,
                    ))
                    return AdmmStatus.GLOBAL_CONVERGENCE

        self.output.put(AdmmIterationResult(
            self.ID,
            nu,
            start,
            stop,
            primalgap,
            dualgap,
            self.max_mismatch(), # was updated in last call to `_receive` # TODO: rename
            objective,
            penalty,
        ))

        if nu == self.admmopt.max_iter - 1:
            self._log('Timeout: Terminating.')
            self._set_status(AdmmStatus.TERMINATED)
            self._send_status()
            return AdmmStatus.TERMINATED

        self.model.update_lamda()
        self.model.update_flow_global({k: self.messages[k] for k in self.updated})
        self._update_mismatch((k for k in self.neighbors if k in self.messages))
        self.model.update_rho([msg.rho for msg in self.messages.values()])

        self.model.update_cost_rule()


    def _receive(self):
        variable_senders = set()
        status_msgs = {}

        # read accumulated messages from active neighbors
        while not self.receiving_queue.empty():
            msg = self.receiving_queue.get(block=False)
            if isinstance(msg, AdmmVariableMessage):
                self.messages[msg.sender] = msg
                variable_senders.add(msg.sender)
            elif isinstance(msg, AdmmStatusMessage):
                status_msgs[msg.sender] = msg
                if msg.status == AdmmStatus.TERMINATED:
                    self._set_status(AdmmStatus.TERMINATED)
                    self._log('Received termination message')
                    return None
            else:
                raise RuntimeError(f'Received unexpected item of type: {type(msg)}')

        senders = variable_senders.union(status_msgs.keys())

        self.updated = self.updated.union(variable_senders)
        if variable_senders:
            self._update_mismatch(variable_senders)

        if status_msgs:
            self._update_status(status_msgs)

        return senders


    def _send_variables(self):
        """
        Send an `AdmmMessage` with the current variables to all neighbors.
        """

        for k, que in self.neighbor_queues.items():
            msg = AdmmVariableMessage(
                sender = self.ID,
                flow = self.model.flows_with_neighbor[k],
                lamda = self.model.lamda[
                    self.model.lamda.index.isin(self.model.flows_with_neighbor[k].index)
                ],
                rho = self.model.rho,
                flow_global = self.model.flow_global.loc[
                    self.model.flow_global.index.isin(
                        self.model.flows_with_neighbor[k].index
                    )
                ],
            )
            que.put(msg)


    def _send_status(self):
        """
        Send an `AdmmStatusMessage` with the current status to all other clusters.
        """
        msg = AdmmStatusMessage(
            sender = self.ID,
            status = self.get_status()
        )
        for que in self.sending_queues.values():
            que.put(msg)

        self.status_update = False


    def _update_status(self, status_msgs):
        """
        Update `self.status` according to the received `status_msgs`, then call
        `_check_global_convergence`.
        """
        for k, msg in status_msgs.items():
            self.status[k] = msg.status

        self._check_global_convergence()


    def _check_global_convergence(self):
        """
        Check whether global convergence has been gained or lost and update `self.status`
        accordingly.
        """
        if all(s in [AdmmStatus.LOCAL_CONVERGENCE, AdmmStatus.GLOBAL_CONVERGENCE]
               for s in self.status):
            self._set_status(AdmmStatus.GLOBAL_CONVERGENCE)

        if self.get_status() == AdmmStatus.GLOBAL_CONVERGENCE and any(
            s not in [AdmmStatus.LOCAL_CONVERGENCE, AdmmStatus.GLOBAL_CONVERGENCE]
            for s in self.status):
            self._set_status(AdmmStatus.LOCAL_CONVERGENCE)


    def _update_mismatch(self, senders):
        """
        Update the mismatch convergence status with `senders` to reflect the information
        received in new messages.

        If mismatch convergence with all neighbors is gained or lost, call
        `self._update_convergence()`.
        """
        old_value = all(
            mismatch < self.admmopt.mismatch_tolerance
            for mismatch in self.mismatches.values()
        )

        for k in senders:
            self.mismatches[k] = self._calc_mismatch(k)

        new_value = all(
            mismatch < self.admmopt.mismatch_tolerance
            for mismatch in self.mismatches.values()
        )
        if new_value != old_value:
            self._update_convergence()


    def _calc_mismatch(self, k):
        """
        Calculate the current mismatch gap with neighbor `k`.
        """
        msg = self.messages[k]
        flow_global_with_k = self.model.flow_global.loc[
            self.model.flow_global.index.isin(
                self.model.flows_with_neighbor[k].index
            )
        ]
        raw_mismatch_gap = norm(flow_global_with_k - msg.flow_global)

        if self.admmopt.tolerance_mode == 'absolute':
            mismatch_gap = raw_mismatch_gap / min(1, len(self.model.flow_global))

        elif self.admmopt.tolerance_mode == 'relative':
            normalizer = max(norm(flow_global_with_k), norm(msg.flow_global))
            if normalizer == 0:
                normalizer = 1
            mismatch_gap = norm(flow_global_with_k - msg.flow_global) / normalizer

        return mismatch_gap


    def _update_convergence(self):
        """
        Check for local convergence, i.e. if primal gap, and constraint mismatch
        with all neighbors are below their respective tolerance thresholds.

        Call `_check_global_convergence` if necessary.

        Return the new value.
        """
        old_value = self.local_convergence()
        primal = self.model.primalgap < self.admmopt.primal_tolerance
        mismatch = all(
            mismatch < self.admmopt.mismatch_tolerance
            for mismatch in self.mismatches.values()
        )
        new_value = primal and mismatch

        if new_value != old_value:
            if new_value:
                self._set_status(AdmmStatus.LOCAL_CONVERGENCE)
                self._check_global_convergence()
            else:
                self._set_status(AdmmStatus.NO_CONVERGENCE)

        return new_value


    def _create_model(
        self,
        data_all,
        timesteps,
        dt,
        objective,
        year,
        initial_values,
        sites,
        shared_lines,
        internal_lines,
        cluster_from,
        cluster_to,
        neighbor_cluster,
        hoursPerPeriod=None,
        weighting_order=None,
        threads=None,
        ) -> UrbsAdmmModel:

        self._log('Creating model')
        index = shared_lines.index.to_frame()

        flow_global = pd.Series({
            (t, year, source, target): initial_values.flow_global
            for t in timesteps[1:]
            for source, target in zip(index['Site In'], index['Site Out'])
        })

        flow_global.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)

        lamda = pd.Series({
            (t, year, source, target): initial_values.lamda
            for t in timesteps[1:]
            for source, target in zip(index['Site In'], index['Site Out'])
        })
        lamda.rename_axis(['t', 'stf', 'sit', 'sit_'], inplace=True)

        self._log('Creating urbs model')
        model = urbs.model.create_model(
            data_all,
            timesteps,
            dt,
            objective,
            sites=sites,
            shared_lines=shared_lines,
            internal_lines=internal_lines,
            flow_global=flow_global,
            lamda=lamda,
            rho=self.admmopt.rho,
            hoursPerPeriod=hoursPerPeriod,
            weighting_order=weighting_order,
        )

        # enlarge shared_lines (copies of slices of data_all['transmission'])
        shared_lines['cluster_from'] = cluster_from
        shared_lines['cluster_to'] = cluster_to
        shared_lines['neighbor_cluster'] = neighbor_cluster

        self._log('Creating UrbsAdmmModel')
        input('Continue?')

        return UrbsAdmmModel(
            self.admmopt,
            flow_global,
            lamda,
            model,
            self.neighbors,
            shared_lines,
            index,
            threads=threads,
        )


    def _log(self, *args):
        msg = f'Process[{self.ID}] {" ".join(str(arg) for arg in args)}'
        print(msg)
