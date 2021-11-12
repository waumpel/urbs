from math import ceil
from multiprocessing import Queue
from time import sleep
from typing import Dict, Iterable, List, Set, Union
import urbs
from urbs.admm_async.admm_option import AdmmOption

import numpy as np

from urbs.admm_async.admm_messages import (
    AdmmIterationResult, AdmmVariableMessage, AdmmStatusMessage, AdmmStatus
)
from .admm_model_persistent import AdmmModelPersistent


class AdmmWorker:
    """
    Manages a single `AdmmModelPersistent` in parallel ADMM.

    Attributes:
        - `ID`: ID of this worker. Same as `self.model.ID`.
        - `log_prefix`: A string that is prefixed to all log messages.
        - `result_dir`: Output directory
        - `output`: `mp.Queue` for sending objects to the parent process.
        - `admmopt`: `AdmmOption` object
        - `n_clusters`: Number of clusters in the ADMM algorithm.
        - `neighbors`: Iterable of neighboring clusters.
        - `n_neighbors`: Number of neighboring clusters.
        - `n_wait`: Number of neighbors for which to wait before starting a new iteration.
        - `receiving_queue`: `mp.Queue` for receiving objects from other workers and the
          parent process.
        - `sending_queues`: Dict of queues for sending objects to other workers.
        - `neighbor_queues`: Dict of queues for sending objects to neighboring workers.
        - `messages`: Dict of the most recent messages received from other workers.
        - `updated`: Set of neighboring clusters that have sent variable updates since the
          start of the current local iteration.
        - `status`: `List[AdmmStatus]` storing the current status of all clusters.
        - `status_update`: Flag indicating that this worker's status has been changed since
          the last status message was sent.
        - `mismatches`: Dict of constraint mismatches with all neighbors.
        - `model`: An `AdmmModelPersistent`.
    """

    def __init__(
        self,
        ID: int,
        result_dir: str,
        output: Queue,
        admmopt: AdmmOption,
        n_clusters: int,
        neighbors: Iterable[str],
        queues: List[Queue]
        ) -> None:

        self.ID: int = ID
        self.log_prefix: str = f'AdmmWorker[{ID}]'
        self.result_dir = result_dir
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
        self.model: AdmmModelPersistent = None


    def get_status(self) -> AdmmStatus:
        """
        Return this worker's current status.
        """
        return self.status[self.ID]


    def _set_status(self, value: AdmmStatus):
        """
        Set this worker's status to value and set `status_update` if the value has changed.
        Refuse status changes if the current status is `TERMINATED`.
        """
        old_value = self.get_status()
        if old_value != AdmmStatus.TERMINATED and old_value != value:
            self.status[self.ID] = value
            self.status_update = True


    def local_convergence(self) -> bool:
        """
        Return the current local convergence status.
        Local convergence is reached if primal gap and constraint mismatch with
        all neighbors are below their respective tolerance thresholds.
        """
        return self.get_status() in [
            AdmmStatus.LOCAL_CONVERGENCE, AdmmStatus.GLOBAL_CONVERGENCE
        ]


    def all_converged(self) -> bool:
        """
        Return whether this cluster and all neighbors who have sent a variable update during
        the current iteration have converged.
        """
        return self.local_convergence() and all(
            self.status[k] in [AdmmStatus.LOCAL_CONVERGENCE, AdmmStatus.GLOBAL_CONVERGENCE]
            for k in self.updated)


    def all_global_convergence(self) -> bool:
        """
        Return whether all clusters have reached global convergence.
        """
        return all(s == AdmmStatus.GLOBAL_CONVERGENCE for s in self.status)


    def max_mismatch(self) -> float:
        """
        Return the maximum mismatch value with all neighbors.
        """
        return np.nan if np.nan in self.mismatches.values() else \
               float(max(self.mismatches.values()))


    @staticmethod
    def run_worker(
        ID,
        result_dir,
        output,
        data_all,
        timesteps,
        dt,
        objective,
        admmopt,
        n_clusters,
        sites,
        neighbors,
        internal_lines,
        shared_lines,
        shared_lines_index,
        flow_global,
        lamda,
        queues,
        hoursPerPeriod=None,
        weighting_order=None,
        threads=None,
        ) -> None:
        """
        Create an `AdmmWorker` and call its `run` method.

        Intended as the target function for `mp.Process`es.
        """
        worker = AdmmWorker(ID, result_dir, output, admmopt, n_clusters, neighbors, queues)
        worker.run(
            data_all,
            timesteps,
            dt,
            objective,
            sites,
            internal_lines,
            shared_lines,
            shared_lines_index,
            flow_global,
            lamda,
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
        sites,
        internal_lines,
        shared_lines,
        shared_lines_index,
        flow_global,
        lamda,
        hoursPerPeriod=None,
        weighting_order=None,
        threads=None,
        ) -> None:
        """
        Start this `AdmmWorker`.

        Create an `AdmmModelPersistent`, then start solving ADMM iterations once the start signal
        has been received from the parent process.
        """
        self._log('Creating model')

        urbs_model = urbs.model.create_model(
            data_all,
            timesteps,
            dt,
            objective,
            sites=sites,
            shared_lines=shared_lines,
            internal_lines=internal_lines,
            hoursPerPeriod=hoursPerPeriod,
            weighting_order=weighting_order,
        )

        self.model = AdmmModelPersistent(
            self.ID,
            self.result_dir,
            self.admmopt,
            urbs_model,
            self.neighbors,
            shared_lines,
            shared_lines_index,
            flow_global,
            lamda,
            threads=threads,
        )

        self._log('Model created')
        self.output.put({'sender': self.ID, 'msg': 'model created'})

        msg = self.receiving_queue.get(block=True)
        if msg != 'start solving':
            self._log('Did not receive start signal, terminating')
            return

        self._log(f'Starting subproblem for regions {sites}.')

        for nu in range(self.admmopt.max_iter):
            result = self._run_iteration(nu)
            if result in [AdmmStatus.TERMINATED, AdmmStatus.GLOBAL_CONVERGENCE]:
                break

        self.output.put(AdmmStatusMessage(self.ID, self.get_status()))


    def _run_iteration(self, nu: int) -> Union[None, AdmmStatus]:
        """
        Run a single ADMM iteration.

        Args:
            - `nu`: Current local iteration counter.

        Return:
            `AdmmStatus.TERMINATED` or `AdmmStatus.GLOBAL_CONVERGENCE` if appropriate,
            otherwise `None`.
        """

        self.updated = set()

        objective, primalgap, dualgap, penalty, start, stop = self.model.solve_iteration()
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
                self.max_mismatch(), # was updated in last call to `_receive`
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
                        self.max_mismatch(), # was updated in last call to `_receive`
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
            self.max_mismatch(), # was updated in last call to `_receive`
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
        # update mismatch again after updating our own flow_global values
        self._update_mismatch((k for k in self.neighbors if k in self.messages))
        self.model.update_rho([msg.rho for msg in self.messages.values()])


    def _receive(self) -> Union[Set[int], AdmmStatus]:
        """
        Check for new messages from other workers.

        Update constraint mismatches with neighbors who have sent variable updates.
        Update status.

        Return `AdmmStatus.TERMINATED` if appropriate, otherwise a set of IDs of all workers
        who have sent a message.
        """
        variable_senders = set()
        status_msgs = {}

        # read accumulated messages from neighbors
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


    def _send_variables(self) -> None:
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


    def _send_status(self) -> None:
        """
        Send an `AdmmStatusMessage` with the current status to all other clusters.

        Set the `status_update` flag to `False`.
        """
        msg = AdmmStatusMessage(
            sender = self.ID,
            status = self.get_status()
        )
        for que in self.sending_queues.values():
            que.put(msg)

        self.status_update = False


    def _update_status(self, status_msgs: Dict) -> None:
        """
        Update `self.status` according to the received `status_msgs`, then call
        `_check_global_convergence`.
        """
        for k, msg in status_msgs.items():
            self.status[k] = msg.status

        self._check_global_convergence()


    def _check_global_convergence(self) -> None:
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


    def _update_mismatch(self, senders: Iterable[int]) -> None:
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
            self.mismatches[k] = self.model.mismatch(k, self.messages[k].flow_global)

        new_value = all(
            mismatch < self.admmopt.mismatch_tolerance
            for mismatch in self.mismatches.values()
        )
        if new_value != old_value:
            self._update_convergence()


    def _update_convergence(self) -> bool:
        """
        Check for local convergence, i.e. if primal gap, and constraint mismatch
        with all neighbors are below their respective tolerance thresholds.

        Call `_check_global_convergence` if necessary.

        Return the new value.
        """
        old_value = self.local_convergence()
        primal = (self.admmopt.primal_tolerance is None or
                  self.model.primalgap < self.admmopt.primal_tolerance)
        dual = (self.admmopt.dual_tolerance is None or
                self.model.dualgap < self.admmopt.dual_tolerance)
        mismatch = self.admmopt.mismatch_tolerance is None or all(
            mismatch < self.admmopt.mismatch_tolerance
            for mismatch in self.mismatches.values()
        )
        new_value = primal and dual and mismatch

        if new_value != old_value:
            if new_value:
                self._set_status(AdmmStatus.LOCAL_CONVERGENCE)
                self._check_global_convergence()
            else:
                self._set_status(AdmmStatus.NO_CONVERGENCE)

        return new_value


    def _log(self, *args) -> None:
        msg = f'Process[{self.ID}] {" ".join(str(arg) for arg in args)}'
        print(msg)
