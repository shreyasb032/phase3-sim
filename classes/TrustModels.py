from typing import Dict
from numpy.random import default_rng
from classes.PerformanceMetrics import PerformanceMetricBase
from classes.State import HumanInfo, Observation


class TrustModelBase:
    """
    Base class for a trust dynamics model
    """

    def __init__(self):
        pass

    def update_trust(self, info: HumanInfo, obs: Observation, wh: float):
        """
        Updates and returns the trust level of the human on the robot.
        Must be implemented by a child class
        :param info: the information available to the human at the time of decision-making
        :param obs: the observation of the outcome of action selection
        :param wh: the health reward weight of the human
        """
        raise NotImplementedError


class BetaDistributionModel(TrustModelBase):
    """
    Maintains and updates parameters of a beta distribution to model trust.
    Uses the complete performance history to update trust
    Guo et al. (2021) - Modeling and Predicting Trust Dynamics in Human–Robot Teaming: A Bayesian Inference Approach
    """

    def __init__(self, parameters: Dict[str, float], performance_metric: PerformanceMetricBase,
                 seed: int | None = None):
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
        self.performance_metric = performance_metric

        self.alpha = self.parameters['alpha0']
        self.beta = self.parameters['beta0']

        self.trust_mean = self.alpha / (self.alpha + self.beta)
        self.trust_sampled = self.rng.beta(self.alpha, self.beta)

    def update_trust(self, info: HumanInfo, obs: Observation, wh: float):
        """
        Updates and returns the trust level of the human on the robot.
        Must be implemented by a child class
        :param info: the information available to the human at the time of decision-making
        :param obs: the observation of the outcome of action selection
        :param wh: the health reward weight of the human
        """
        performance = self.performance_metric.get_performance(info, obs, wh)
        self.num_successes += performance
        self.num_failures += (1 - performance)
        self.performance_history.append(performance)
        self.alpha = self.parameters['alpha0'] + self.num_successes * self.parameters['vs']
        self.beta = self.parameters['beta0'] + self.num_failures * self.parameters['vf']

        self.trust_mean = self.alpha / (self.alpha + self.beta)
        self.trust_sampled = self.rng.beta(self.alpha, self.beta)

    def update_parameters(self, parameters: Dict[str, float]):
        self.parameters = parameters
        self.alpha = self.parameters['alpha0'] + self.num_successes * self.parameters['vs']
        self.beta = self.parameters['beta0'] + self.num_failures * self.parameters['vf']
        self.trust_mean = self.alpha / (self.alpha + self.beta)
        self.trust_sampled = self.rng.beta(self.alpha, self.beta)

    def get_performance(self, info: HumanInfo, obs: Observation, wh: float):
        return self.performance_metric.get_performance(info, obs, wh)
