from classes.RewardModels import RewardModelBase
from classes.TrustModels import BetaDistributionModel
from classes.DecisionModels import BoundedRationalityDisuse


class Human:

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

    def update_trust(self, rec, threat, threat_level, health=100, time=0):
        """Update trust based on immediate observed reward
        :param rec: recommendation given to the human
        :param threat: integer representing the presence of threat
        :param threat_level: a float representing the level of threat
        :param health: the current health level of the soldier
        :param time: the time spent in the mission"""

        performance = 0
        if rec == threat:
            performance = 1
        self.trust_model.update(performance)

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
