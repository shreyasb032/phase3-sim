from numpy.random import default_rng
from scipy.special import expit
from classes.State import HumanInfo
from classes.RewardModels import RewardModelBase


class DecisionModelBase:
    """
    The base class for a human Decision Model
    """

    def __init__(self, seed: int | None = None):
        """
        :param seed: the seed for the random number generator (default: 123)
        """
        self.rng = default_rng(seed)

    def choose_action(self, info: HumanInfo, trust: float, **kwargs):
        """
        Chooses an action given the recommendation, the trust, and the set of possible actions
        :param info: The information available to the human
        :param trust: The trust of the human on the robot
        :param kwargs: additional keyword arguments associated with specific decision models
        :return:
        """
        raise NotImplementedError


class BoundedRationalityDisuse(DecisionModelBase):
    """
    The bounded rationality disuse model of human decision-making
    """

    def __init__(self, kappa: float, seed: int | None = None):
        super().__init__(seed)
        self.prob_1 = None
        self.prob_0 = None
        self.reward_1 = None
        self.reward_0 = None
        self.kappa = kappa
        self.wh = None
        self.wc = None
        self.threat_level = None

    def choose_action(self, info: HumanInfo, trust: float, **kwargs) -> int:
        """
        :param info: The information available to the human while making a decision
        :param trust: The trust of the human on the robot
        :param kwargs: additional keyword arguments associated with specific decision models
                        should contain: wh
        :return: Chosen action
        """

        self.wh = kwargs['wh']
        self.wc = 1 - self.wh
        self.threat_level = info.threat_level

        self.reward_0 = -self.threat_level * self.wh
        self.reward_1 = -self.wc
        
        self.prob_0 = expit(self.kappa * (self.reward_0 - self.reward_1))
        self.prob_1 = 1 - self.prob_0        

        r = self.rng.random()
        # With probability equal to trust, return the recommended action
        if r < trust:
            return info.recommendation

        # else, choose according to bounded rationality
        return self.rng.choice([0, 1], p=[self.prob_0, self.prob_1])

    def get_prob_of_actions(self, info: HumanInfo, trust: float, **kwargs):
        """
        Returns the probabilities for the two actions
        :param info: the information available to the human while making the decision
        :param trust: the trust of the human on the robot
        :param kwargs: any other keyword arguments
        :return: (prob_0, prob_1) the tuple of probabilities for choosing the two actions
        """
        _ = self.choose_action(info, trust, **kwargs)
        if info.recommendation == 0:
            prob_0 = trust + (1 - trust) * self.prob_0
            prob_1 = (1 - trust) * self.prob_1
        else:
            prob_0 = (1 - trust) * self.prob_0
            prob_1 = trust + (1 - trust) * self.prob_1

        return prob_0, prob_1
