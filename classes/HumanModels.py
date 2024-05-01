from classes.RewardModels import RewardModelBase
from classes.TrustModels import BetaDistributionModel
from classes.DecisionModels import BoundedRationalityDisuse
from classes.ParamsUpdater import Estimator


class Human:
    """
    This is the class for the simulated human. It has a fixed trust_model, decision_model, and reward_model
    """

    def __init__(self, trust_model: BetaDistributionModel, decision_model: BoundedRationalityDisuse,
                 reward_model: RewardModelBase):
        """
        :param trust_model: the trust dynamics model of this human
        :param decision_model: the decision-making model of this human
        :param reward_model: the reward weights model of this human
        """
        self.trust_model = trust_model
        self.decision_model = decision_model
        self.reward_model = reward_model

    def update_trust(self, recommendation: int, threat: int, threat_level: float, wh: float):
        """Update trust based on immediate observed reward
        :param recommendation: recommendation given to the human
        :param threat: integer representing the presence of threat
        :param threat_level: a float representing the level of threat
        :param wh: the health reward weight of the human
        """
        self.trust_model.update_trust(recommendation, threat, threat_level, wh)

    def get_trust_mean(self):
        """Returns the mean level of trust"""
        return self.trust_model.trust_mean

    def get_trust_sample(self):
        """Samples trust from the beta distribution"""
        return self.trust_model.trust_sampled

    def choose_action(self, rec: int, threat_level: float, health: int, time: int) -> int:
        """
        Chooses an action
        :param rec: the system's recommended action [0 or 1]
        :param threat_level: the level of threat reported by the drone
        :param health: the current health level of the soldier
        :param time: the time remaining to complete the mission
        :return: the chosen action 0 or 1
        """
        trust = self.trust_model.trust_sampled
        wh = self.reward_model.get_wh(health=health, time=time)
        return self.decision_model.choose_action(rec, trust, wh=wh, threat_level=threat_level)


class HumanModel(Human):
    """
    The human model maintained by the robot. This one will have a variable trust dynamics model which is updated
    after getting trust feedback from the simulated human
    """
    def __init__(self, trust_model: BetaDistributionModel, decision_model: BoundedRationalityDisuse,
                 reward_model: RewardModelBase):
        super().__init__(trust_model, decision_model, reward_model)
        self.trust_model_updater = Estimator(self.trust_model)

    def update_trust_model(self, trust_feedback: float, site_idx: int):
        self.trust_model = self.trust_model_updater.update_model(trust_feedback, site_idx)

# class ReversePsychology(HumanBase):
#     """The reverse psychology model of human behavior"""
#
#     def __init__(self, params: List, reward_weights: Dict, reward_fun: RewardsBase,
#                  performance_metric: PerformanceMetricBase):
#         """
#         Initializes this human model
#         :param params: the trust parameters associated with the human. List [alpha0, beta0, ws, wf]
#         :param reward_weights: the reward weights associated with the human. Dict with keys 'health' and 'time'
#         :param reward_fun: the reward function associated with this human. Must have the function reward(health, time)
#         :param performance_metric: the performance metric that returns the performance given the recommendation and
#                                     outcome
#         """
#         super().__init__(params, reward_weights, reward_fun, performance_metric)
#
#     def choose_action(self, rec, threat_level=None, health=None, time=None):
#         """
#         Chooses an action based on the reverse psychology model
#         :param rec: the recommendation given to the human
#         :param threat_level: a float representing the level fo threat in the current site
#         :param health: the current health level of the soldier
#         :param time: the time spent in the mission
#         """
#
#         return np.random.choice([rec, 1 - rec], p=[self.trust, 1 - self.trust])
#
#
# class OneStepOptimal(HumanBase):
#     """Accept recommendation with probability trust, choose the action which gives the best
#         immediate expected reward otherwise"""
#
#     def __init__(self, params: List, reward_weights: Dict, reward_fun: RewardsBase,
#                  performance_metric: PerformanceMetricBase):
#         """
#         Initializes the human base class
#         :param params: the trust parameters associated with the human. List [alpha0, beta0, ws, wf]
#         :param reward_weights: the reward weights associated with the human. Dict with keys 'health' and 'time'
#         :param reward_fun: the reward function associated with this human. Must have the function reward(health, time)
#         :param performance_metric: the performance metric that returns the performance given the recommendation and
#                                     outcome
#         """
#         super().__init__(params, reward_weights, reward_fun, performance_metric)
#
#     def choose_action(self, rec, threat_level, health, time):
#         """
#         Chooses an action based on the disuse model
#         :param rec: the recommendation given to the human
#         :param threat_level: a float representing the level fo threat in the current site
#         :param health: the current health level of the soldier
#         :param time: the time spent in the mission
#         """
#
#         p = np.random.uniform(0, 1)
#
#         # With probability = trust, choose the recommendation
#         if p < self.trust:
#             return rec
#
#         hl, tc = self.reward_fun.reward(health, time, house=None)
#
#         # With probability = 1 - trust, choose the action that maximizes immediate expected reward
#         r0 = self.reward_weights["health"] * threat_level * hl
#         r1 = self.reward_weights["time"] * tc
#
#         return int(r0 < r1)
#
#
# class BoundedRational(HumanBase):
#     """Bounded rationality with disuse model. Accepts recommendation with probability trust,
#     Chooses the action that gives the best immediate expected reward with probability proportional
#     to the exponent of that reward multiplied by the rationality coefficient"""
#
#     def __init__(self, params, reward_weights, reward_fun: RewardsBase,
#                  performance_metric: PerformanceMetricBase,
#                  kappa=0.05):
#         """
#         Initializes the human base class
#         :param params: the trust parameters associated with the human. List [alpha0, beta0, ws, wf]
#         :param reward_weights: the reward weights associated with the human. Dict with keys 'health' and 'time'
#         :param reward_fun: the reward function associated with this human. Must have the function reward(health, time)
#         :param performance_metric: the performance metric that returns the performance given the recommendation and
#                                     outcome
#         :param kappa: the rationality coefficient for the human
#         """
#         super().__init__(params, reward_weights, reward_fun, performance_metric)
#         self.kappa = kappa
#
#     def choose_action(self, rec, threat_level, health, time):
#         """
#         Chooses an action based on the bounded rationality disuse model
#         :param rec: the recommendation given to the human
#         :param threat_level: a float representing the level fo threat in the current site
#         :param health: the current health level of the soldier
#         :param time: the time spent in the mission
#         """
#
#         p = np.random.uniform(0, 1)
#
#         # With probability = trust, choose the recommendation
#         if p < self.trust:
#             return rec
#
#         # Else, compute the rewards for action 0 and action 1
#         hl, tc = self.reward_fun.reward(health, time, house=None)
#         r0 = self.reward_weights["health"] * threat_level * hl
#         r1 = self.reward_weights["time"] * tc
#
#         p0 = 1. / (1 + np.exp(r1 - r0))
#
#         return np.random.choice([0, 1], p=[p0, 1 - p0])
