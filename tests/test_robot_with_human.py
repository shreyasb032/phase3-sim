import _context
from classes.RobotModel import Robot
from classes.HumanModels import Human, HumanModel
from classes.TrustModels import BetaDistributionModel
from classes.DecisionModels import BoundedRationalityDisuse
from classes.Simulation import SimSettings
from classes.PerformanceMetrics import ObservedReward
from classes.RewardModels import ConstantWeights, StateDependentWeights
from classes.State import RobotInfo, HumanInfo
from numpy.random import default_rng
from copy import deepcopy


rng = default_rng()

num_sites = 5
start_health = 100
start_time = 100
prior_threat_level = 0.7
discount_factor = 0.6
threat_seed = 123

settings = SimSettings(num_sites, start_health, start_time, prior_threat_level,
                       discount_factor, threat_seed)

wh = 0.87
reward_model1 = ConstantWeights(wh=wh)
reward_model2 = StateDependentWeights()

parameters = {'alpha0': 10., 'beta0': 50., 'vs': 10., 'vf': 20.}
performance_metric = ObservedReward()
trust_model = BetaDistributionModel(parameters, performance_metric, seed=rng.integers(low=0, high=20))
decision_model = BoundedRationalityDisuse(kappa=0.1, seed=rng.integers(low=0, high=20))
human = Human(trust_model, decision_model, reward_model2)

trust_model_robot = deepcopy(trust_model)
for k, v in trust_model_robot.parameters.items():
    trust_model_robot.parameters[k] = v + 5

human_model = HumanModel(trust_model_robot, decision_model, reward_model1)
robot1 = Robot(human_model, reward_model1, settings)
robot2 = Robot(human_model, reward_model2, settings)

health = start_health
time = start_time

d_1_star = (1 - wh)/wh

fake_human_info = HumanInfo(health, time, d_1_star, -1, 0)
wh_sd = reward_model2.get_wh(fake_human_info)
d_2_star = (1 - wh_sd) / wh_sd

# threat_level = settings.threat_setter.after_scan[0]
threat_level = rng.uniform(low=d_1_star, high=d_2_star)
info = RobotInfo(health, time, threat_level, settings.d, 0)
rec1 = robot1.get_recommendation(info)
rec2 = robot2.get_recommendation(info)

print(rf"$d_1^*$: {d_1_star}, $d_2^*$: {d_2_star}")
print(f"Threat level {round(threat_level, 2)}, rec1 {rec1}, rec2 {rec2}")
