import _context
from classes.HumanModels import Human
from classes.TrustModels import BetaDistributionModel
from classes.PerformanceMetrics import ObservedReward
from classes.DecisionModels import BoundedRationalityDisuse
from classes.RewardModels import ConstantWeights
from numpy.random import default_rng
from classes.State import HumanInfo, Observation

rng = default_rng()

parameters = {'alpha0': 120, 'beta0': 50., 'vs': 10., 'vf': 20.}
performance_metric = ObservedReward()
trust_model = BetaDistributionModel(parameters, performance_metric, seed=rng.integers(low=0, high=20))
decision_model = BoundedRationalityDisuse(kappa=0.2, seed=rng.integers(low=0, high=20))
reward_model = ConstantWeights(wh=1.0)

human = Human(trust_model, decision_model, reward_model)

recommendation = 0
threat = 1
threat_level = 0.8
health = 90
time = 90

info = HumanInfo(health, time, threat_level, recommendation, site_idx=0)
action = human.choose_action(info)
print(f"Chosen action: {action}")
observation = Observation(threat, action)

# Testing mean trust
trust_mean = human.get_trust_mean()
print(f"Trust mean before update: {trust_mean:.2f}")

# Testing sampled trust
trust_sample = human.get_trust_sample()
print(f"Trust sampled before update: {trust_sample:.2f}")

# Testing update_trust
print()
print("Updating trust...")
human.update_trust(info, observation, reward_model.get_wh(info))
trust_mean = human.get_trust_mean()
print(f"Trust mean after update: {trust_mean:.2f}")

trust_sample = human.get_trust_sample()
observation.add_trust_feedback(trust_sample)
print(f"Trust sampled after update: {trust_sample:.2f}")
