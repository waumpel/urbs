from warnings import warn

class AdmmOption(object):
    def __init__(self,
        primal_tolerance,
        dual_tolerance,
        mismatch_tolerance,
        rho,
        max_penalty = None,
        penalty_mult = None,
        primal_decrease = None,
        residual_distance = None,
        async_correction = 0,
        wait_percent = 0.01,
        wait_time = 0.1,
        max_iter = 1000,
        tolerance_mode = 'absolute',
        penalty_mode = None,
    ):
        if tolerance_mode not in ['absolute', 'relative']:
            raise ValueError("tolerance_mode must be 'absolute' or 'relative'")

        if penalty_mode is None:
            if (max_penalty is None and
                penalty_mult is None and
                primal_decrease is None and
                residual_distance is None):
                penalty_mode = 'fixed'
            elif (max_penalty is not None and
                  penalty_mult is not None and
                  primal_decrease is not None and
                  residual_distance is None):
                  penalty_mode = 'increasing'
            elif (max_penalty is not None and
                  penalty_mult is not None and
                  primal_decrease is None and
                  residual_distance is not None):
                  penalty_mode = 'residual_balancing'
            else:
                raise ValueError("Cannot infer penalty_mode")

        if penalty_mode not in ['fixed', 'increasing', 'residual_balancing']:
            raise ValueError("tolerance_mode must be 'fixed', 'increasing' or 'residual_balancing'")

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
            elif max_penalty <= rho:
                raise ValueError("max_penalty must be larger than rho")
            if penalty_mult is None:
                penalty_mult = 1.1
            elif penalty_mult <= 1:
                raise ValueError("penalty_mult must be larger than 1")
            if residual_distance is None:
                residual_distance = 1.1
            elif residual_distance <= 0:
                raise ValueError("residual_distance must be larger than 0")


        if primal_tolerance <= 0:
            raise ValueError("primal_tolerance must be larger than 0")
        if dual_tolerance <= 0:
            raise ValueError("dual_tolerance must be larger than 0")
        if mismatch_tolerance <= 0:
            raise ValueError("mismatch_tolerance must be larger than 0")
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

        self.primal_tolerance = primal_tolerance
        self.dual_tolerance = dual_tolerance
        self.mismatch_tolerance = mismatch_tolerance
        self.rho = rho
        self.max_penalty = max_penalty
        self.penalty_mult = penalty_mult
        self.primal_decrease = primal_decrease
        self.residual_distance = residual_distance
        self.async_correction = async_correction
        self.wait_percent = wait_percent
        self.wait_time = wait_time
        self.max_iter = max_iter
        self.tolerance_mode = tolerance_mode
        self.penalty_mode = penalty_mode

