import _context
from classes.HumanModels import Human
from classes.TrustModels import BetaDistributionModel
from classes.PerformanceMetrics import ObservedReward
from classes.DecisionModels import BoundedRationalityDisuse
from classes.RewardModels import ConstantWeights
from numpy.random import default_rng
from classes.State import HumanInfo, Observation

rng = default_rng()

parameters = {'alpha0': 10., 'beta0': 50., 'vs': 10., 'vf': 20.}
performance_metric = ObservedReward()
trust_model = BetaDistributionModel(parameters, performance_metric, seed=rng.integers(low=0, high=20))
decision_model = BoundedRationalityDisuse(kappa=0.1, seed=rng.integers(low=0, high=20))
reward_model = ConstantWeights(wh=0.95)

human = Human(trust_model, decision_model, reward_model)

recommendation = 1
threat = 1
threat_level = 0.8
health = 90
time = 90

info = HumanInfo(health, time, threat_level, recommendation)
observation = Observation(threat)

# Testing action choice
# print(human.choose_action(info))

# Testing mean trust
# print(human.get_trust_mean())

# Testing sampled trust
# print(human.get_trust_sample())

# Testing update_trust
human.update_trust(info, observation, reward_model.get_wh(info))
print(human.get_trust_mean())
observation.add_trust_feedback(human.get_trust_sample())
print(observation.trust_feedback)
