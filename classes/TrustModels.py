from typing import Dict
from numpy.random import default_rng
from classes.PerformanceMetrics import PerformanceMetricBase


class TrustModelBase:
    """
    Base class for a trust dynamics model
    """

    def __init__(self):
        pass

    def update_trust(self, recommendation: int, threat: int, threat_level: float, wh: float):
        """
        Updates and returns the trust level of the human on the robot.
        Must be implemented by a child class
        :param recommendation: the recommended action of the system
        :param threat: an integer representing the observed presence of threat inside the site
        :param threat_level: the threat level reported by the drone
        :param wh: the health reward weight of the human
        """
        raise NotImplementedError


class BetaDistributionModel(TrustModelBase):
    """
    Maintains and updates parameters of a beta distribution to model trust.
    Uses the complete performance history to update trust
    Guo et al. (2021) - Modeling and Predicting Trust Dynamics in Human–Robot Teaming: A Bayesian Inference Approach
    """

    def __init__(self, parameters: Dict[str, float], performance_metric: PerformanceMetricBase, seed: int = 123):
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

        self.alpha_0 = self.parameters['alpha0']
        self.beta_0 = self.parameters['beta0']
        self.vs = self.parameters['vs']
        self.vf = self.parameters['vf']

        self.alpha = self.alpha_0
        self.beta = self.beta_0

        self.trust_mean = self.alpha / (self.alpha + self.beta)
        self.trust_sampled = self.rng.beta(self.alpha, self.beta)

    def update_trust(self, recommendation: int, threat: int, threat_level: float, wh: float):
        """
        Updates the mean and sampled trust
        :param recommendation: the recommended action of the system
        :param threat: an integer representing the observed presence of threat inside the site
        :param threat_level: the threat level reported by the drone
        :param wh: the health reward weight of the human
        """
        performance = self.performance_metric.get_performance(recommendation, threat, threat_level, wh)
        self.num_successes += performance
        self.num_failures += (1 - performance)
        self.performance_history.append(performance)
        self.alpha = self.alpha_0 + self.num_successes * self.vs
        self.beta = self.beta_0 + self.num_failures * self.vf

        self.trust_mean = self.alpha / (self.alpha + self.beta)
        self.trust_sampled = self.rng.beta(self.alpha, self.beta)

    def update_parameters(self, parameters: Dict[str, float]):
        self.parameters = parameters

    def get_performance(self, recommendation: int, threat: int, threat_level: float, wh: float):
        return self.performance_metric.get_performance(recommendation, threat, threat_level, wh)
