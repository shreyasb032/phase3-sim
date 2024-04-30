from typing import Dict
from numpy.random import default_rng


class TrustModelBase:
    """
    Base class for a trust dynamics model
    """

    def __init__(self):
        pass

    def update(self, performance: int):
        """
        Updates and returns the trust level of the human on the robot.
        Must be implemented by a child class
        :param performance: the performance of the system
        """
        raise NotImplementedError


class BetaDistributionModel(TrustModelBase):
    """
    Maintains and updates parameters of a beta distribution to model trust.
    Uses the complete performance history to update trust
    Guo et al. (2021) - Modeling and Predicting Trust Dynamics in Human–Robot Teaming: A Bayesian Inference Approach
    """

    def __init__(self, parameters: Dict[str, float], seed: int = 123):
        """
        Initializes the class
        :param parameters: a dict with keys alpha0, beta0, vs, vf and corresponding values
        :param seed (optional): the seed to start the random number generator (default: 123)
        """
        super().__init__()
        self.parameters = parameters
        self.rng = default_rng(seed=seed)

        self.performance_history = []
        self.num_successes = 0
        self.num_failures = 0

        self.alpha_0 = self.parameters['alpha0']
        self.beta_0 = self.parameters['beta0']
        self.vs = self.parameters['vs']
        self.vf = self.parameters['vf']

        self.alpha = self.alpha_0
        self.beta = self.beta_0

        self.trust_mean = self.alpha / (self.alpha + self.beta)
        self.trust_sampled = self.rng.beta(self.alpha, self.beta)

    def update(self, performance: int):
        """
        Updates the mean and sampled trust
        :param performance: the performance of the system
        """
        self.num_successes += performance
        self.num_failures += (1 - performance)
        self.performance_history.append(performance)
        self.alpha = self.alpha_0 + self.num_successes * self.vs
        self.beta = self.beta_0 + self.num_failures * self.vf

        self.trust_mean = self.alpha / (self.alpha + self.beta)
        self.trust_sampled = self.rng.beta(self.alpha, self.beta)
