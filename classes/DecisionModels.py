from typing import Set
from numpy.random import default_rng
from scipy.special import expit


class DecisionModelBase:
    """
    The base class for a human Decision Model
    """

    def __init__(self, seed: int = 123):
        """
        :param seed: the seed for the random number generator (default: 123)
        """
        self.rng = default_rng(seed)

    def choose_action(self, recommendation: int, trust: float, **kwargs):
        """
        Chooses an action given the recommendation, the trust, and the set of possible actions
        :param recommendation: The recommended action by the system
        :param trust: The trust of the human on the robot
        :param kwargs: additional keyword arguments associated with specific decision models
        :return:
        """
        raise NotImplementedError


class BoundedRationalityDisuse(DecisionModelBase):
    """
    The bounded rationality disuse model of human decision-making
    """

    def __init__(self, kappa: float, seed: int = 123):
        super().__init__(seed)
        self.prob_1 = None
        self.prob_0 = None
        self.reward_1 = None
        self.reward_0 = None
        self.kappa = kappa
        self.wh = None
        self.wc = None
        self.threat_level = None

    def choose_action(self, recommendation: int, trust: float, **kwargs) -> int:
        """
        :param recommendation: The recommended action by the system
        :param trust: The trust of the human on the robot
        :param kwargs: additional keyword arguments associated with specific decision models
                        should contain: wh, threat_level
        :return: Chosen action
        """

        r = self.rng.random()
        # With probability equal to trust, return the recommended action
        if r < trust:
            return recommendation

        # else, choose according to bounded rationality

        self.wh = kwargs['wh']
        self.wc = 1 - self.wh
        self.threat_level = kwargs['threat_level']

        self.reward_0 = -self.threat_level * self.wh
        self.reward_1 = -self.wc
        
        self.prob_0 = expit(self.kappa * (self.reward_0 - self.reward_1))
        self.prob_1 = 1 - self.prob_0        
        
        return self.rng.choice([0, 1], p=[self.prob_0, self.prob_1])
