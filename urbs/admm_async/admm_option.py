from typing import Tuple, Union
from warnings import warn

class AdmmOption(object):
    def __init__(self,
        rho,
        scale_rho = False,
        penalty_mode = None,
        max_penalty = None,
        penalty_mult = None,
        primal_decrease = None,
        residual_distance = None,
        mult_adapt = None,
        max_mult = None,
        async_correction = 0,
        wait_percent = 0.01,
        wait_time = 0.1,
        max_iter = 1000,
        tolerance: Union[float, Tuple[float, float, float]] = 0.01,
        ):

        if penalty_mode is None:
            if (max_penalty is None and
                penalty_mult is None and
                primal_decrease is None and
                residual_distance is None and
                mult_adapt is None and
                max_mult is None):
                penalty_mode = 'fixed'
            elif (max_penalty is not None and
                penalty_mult is not None and
                primal_decrease is not None and
                residual_distance is None and
                mult_adapt is None and
                max_mult is None):
                penalty_mode = 'increasing'
            elif (max_penalty is not None and
                penalty_mult is not None and
                primal_decrease is None and
                residual_distance is not None and
                mult_adapt is None and
                max_mult is None):
                penalty_mode = 'residual_balancing'
            elif (max_penalty is not None and
                penalty_mult is None and
                primal_decrease is None and
                residual_distance is not None and
                mult_adapt is not None and
                max_mult is not None):
                penalty_mode = 'adaptive_multiplier'
            else:
                raise ValueError("Cannot infer penalty_mode")

        if penalty_mode not in ['fixed', 'increasing', 'residual_balancing', 'adaptive_multiplier']:
            raise ValueError("penalty_mode must be 'fixed', 'increasing', 'residual_balancing' or 'adaptive_multiplier'")

        if penalty_mode == 'fixed':
            if max_penalty is not None:
                warn("max_penalty will be ignored because penalty_mode == 'fixed'")
            if penalty_mult is not None:
                warn("penalty_mult will be ignored because penalty_mode == 'fixed'")
            if primal_decrease is not None:
                warn("primal_decrease will be ignored because penalty_mode == 'fixed'")

        elif penalty_mode == 'increasing':
            if max_penalty is None:
                raise ValueError("max_penalty is required when using penalty_mode 'increasing'")
            elif max_penalty <= rho:
                raise ValueError("max_penalty must be larger than rho")
            if penalty_mult is None:
                penalty_mult = 1.1
            elif penalty_mult <= 1:
                raise ValueError("penalty_mult must be larger than 1")
            if primal_decrease is None:
                primal_decrease = 0.9
            elif primal_decrease <= 0 or primal_decrease > 1:
                raise ValueError("primal_decrease must be within (0, 1]")

        elif penalty_mode == 'residual_balancing':
            if max_penalty is None:
                raise ValueError("max_penalty is required when using penalty_mode 'residual_balancing'")
            elif max_penalty < rho:
                raise ValueError("max_penalty must be larger than or equal to rho")
            if penalty_mult is None:
                penalty_mult = 1.1
            elif penalty_mult <= 1:
                raise ValueError("penalty_mult must be larger than 1")
            if residual_distance is None:
                residual_distance = 1.1
            elif residual_distance <= 0:
                raise ValueError("residual_distance must be larger than 0")

        elif penalty_mode == 'adaptive_multiplier':
            if max_penalty is None:
                raise ValueError("max_penalty is required when using penalty_mode 'adaptive_multiplier'")
            elif max_penalty < rho:
                raise ValueError("max_penalty must be larger than or equal to rho")
            if residual_distance is None:
                residual_distance = 1
            elif residual_distance <= 0:
                raise ValueError("residual_distance must be larger than 0")
            if mult_adapt is None:
                mult_adapt = 1
            elif mult_adapt <= 0:
                raise ValueError("mult_adapt must be positive")
            if max_mult is None:
                max_mult = 10**2
            elif max_mult <= 1:
                raise ValueError("max_mult must be larger than 1")

        if rho <= 0:
            raise ValueError("rho must be larger than 0")
        if async_correction < 0:
            raise ValueError("async_correction must be larger than or equal to 0")
        if wait_percent <= 0 or wait_percent > 1:
            raise ValueError("wait_percent must be within (0, 1]")
        if wait_time <= 0:
            raise ValueError("wait_time must be larger than 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be larger than 0")

        if isinstance(tolerance, float):
            self.primal_tolerance = tolerance
            self.dual_tolerance = tolerance
            self.mismatch_tolerance = tolerance
        elif isinstance(tolerance, Tuple):
            self.primal_tolerance, self.dual_tolerance, self.mismatch_tolerance = tolerance
        else:
            raise TypeError("tolerance must be float or Tuple[float]")

        if self.primal_tolerance is not None and self.primal_tolerance < 0:
            raise ValueError("primal_tolerance must be non-negative")
        if self.dual_tolerance is not None and self.dual_tolerance < 0:
            raise ValueError("dual_tolerance must be non-negative")
        if self.mismatch_tolerance is not None and self.mismatch_tolerance < 0:
            raise ValueError("mismatch_tolerance must be non-negative")

        self.rho = rho
        self.scale_rho = scale_rho
        self.max_penalty = max_penalty
        self.penalty_mult = penalty_mult
        self.primal_decrease = primal_decrease
        self.residual_distance = residual_distance
        self.mult_adapt = mult_adapt
        self.max_mult = max_mult
        self.async_correction = async_correction
        self.wait_percent = wait_percent
        self.wait_time = wait_time
        self.max_iter = max_iter
        self.penalty_mode = penalty_mode
