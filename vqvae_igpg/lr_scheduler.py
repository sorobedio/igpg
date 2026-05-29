import numpy as np


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n):
        return self.schedule(n)



class CyclicalAnnealingSchedule:
    def __init__(self, cycle_length, beta_min=0.0, beta_max=1.0):
        self.cycle_length = cycle_length  # How many steps in one cycle
        self.beta_min = beta_min  # Minimum value of beta (weight for KL divergence)
        self.beta_max = beta_max  # Maximum value of beta

    def get_beta(self, current_step):
        # Position in the current cycle (0 to 1, where 0 is the start and 1 is the end of the cycle)
        cycle_pos = (current_step % self.cycle_length) / self.cycle_length
        if cycle_pos < 0.5:
            # Increasing phase of the cycle
            return self.beta_min + (self.beta_max - self.beta_min) * (2 * cycle_pos)
        else:
            # Decreasing phase of the cycle
            return self.beta_max - (self.beta_max - self.beta_min) * (2 * (cycle_pos - 0.5))



class TriangularAnnealingScheduler:
    def __init__(self, cycle_length, beta_min=0.0, beta_max=1.0):
        """
        :param cycle_length: Number of steps in a full cycle (from beta_min to beta_max and back to beta_min).
        :param beta_min: Minimum value for beta.
        :param beta_max: Maximum value for beta.
        """
        self.cycle_length = cycle_length
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, current_step):
        """
        Compute the beta value based on the current step.
        :param current_step: The current training step.
        :return: The beta value at this step.
        """
        cycle_position = current_step % self.cycle_length  # Position in the current cycle
        half_cycle = self.cycle_length // 2

        if cycle_position < half_cycle:
            # Increasing phase (first half of the cycle)
            beta = self.beta_min + (self.beta_max - self.beta_min) * (cycle_position / half_cycle)
        else:
            # Decreasing phase (second half of the cycle)
            beta = self.beta_max - (self.beta_max - self.beta_min) * ((cycle_position - half_cycle) / half_cycle)

        return beta



