import _context
from classes.HumanModels import Human, HumanModel
from classes.TrustModels import BetaDistributionModel
from classes.PerformanceMetrics import ObservedReward
from classes.DecisionModels import BoundedRationalityDisuse
from classes.RewardModels import ConstantWeights
from numpy.random import default_rng
from classes.State import HumanInfo, Observation
from copy import deepcopy


rng = default_rng()
parameters = {'alpha0': 10., 'beta0': 50., 'vs': 10., 'vf': 20.}
performance_metric = ObservedReward()
trust_model = BetaDistributionModel(parameters, performance_metric, seed=rng.integers(low=0, high=20))
decision_model = BoundedRationalityDisuse(kappa=0.1, seed=rng.integers(low=0, high=20))
reward_model = ConstantWeights(wh=0.95)

human = Human(trust_model, decision_model, reward_model)

trust_model_robot = deepcopy(trust_model)
for k, v in trust_model_robot.parameters.items():
    trust_model_robot.parameters[k] = v + 5

human_model = HumanModel(trust_model_robot, decision_model, reward_model)
# human_model_cpy = deepcopy(human_model)

# Recommendation is correct
recommendation = 0
threat = 1
threat_level = 0.8
health = 90
time = 90

info = HumanInfo(health, time, threat_level, recommendation, site_idx=0)
action = human.choose_action(info)
observation = Observation(threat, action)

human.update_trust(info, observation, reward_model.get_wh(info))
trust = human.get_trust_mean()

human_model.update_trust(info, observation, reward_model.get_wh(info))
trust_est_before_update = human_model.get_trust_mean()
human_model.update_trust_model(trust)
trust_est_after_update = human_model.get_trust_mean()

print(f"Trust mean: {trust:.2f}")
print(f"Trust estimate before update: {trust_est_before_update:.2f}")
print(f"Trust estimate after update: {trust_est_after_update:.2f}")

print("True trust parameters:", human.trust_model.parameters)
print("Estimated trust parameters:", human_model.trust_model.parameters)
