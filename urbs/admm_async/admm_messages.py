from enum import Enum


class AdmmStatus(Enum):
    NO_CONVERGENCE = 0
    LOCAL_CONVERGENCE = 1
    GLOBAL_CONVERGENCE = 2
    TERMINATED = 3


class AdmmVariableMessage(object):
    def __init__(
        self,
        sender,
        flow,
        lamda,
        rho,
        flow_global,
        ):

        self.sender = sender
        self.flow = flow
        self.lamda = lamda
        self.rho = rho
        self.flow_global = flow_global


class AdmmStatusMessage:
    def __init__(self, sender, status):
        self.sender = sender
        self.status = status


class AdmmIterationResult:

    _HEADERS = [
        'process_id',
        'local_iteration',
        'start_time',
        'stop_time',
        'primalgap',
        'dualgap',
        'mismatch',
        'objective',
        'penalty',
    ]

    def __init__(
        self,
        process_id,
        local_iteration,
        start_time,
        stop_time,
        primalgap,
        dualgap,
        mismatch,
        objective,
        penalty,
        ):

        self.process_id = process_id
        self.local_iteration = local_iteration
        self.start_time = start_time
        self.stop_time = stop_time
        self.primalgap = primalgap
        self.dualgap = dualgap
        self.mismatch = mismatch
        self.objective = objective
        self.penalty = penalty


    def subtract_time(self, time: float):
        self.start_time -= time
        self.stop_time -= time


    def __str__(self) -> str:
        return ' '.join(
            (
                str(attr) for attr in [
                    self.process_id,
                    self.local_iteration,
                    self.start_time,
                    self.stop_time,
                    self.primalgap,
                    self.dualgap,
                    self.mismatch,
                    self.objective,
                    self.penalty,
                ]
            )
        )
