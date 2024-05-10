import _context
from classes.RobotModel import RobotOnly
from classes.Simulation import SimSettings
from classes.RewardModels import ConstantWeights, StateDependentWeights
from classes.State import RobotInfo, HumanInfo
from numpy.random import default_rng


rng = default_rng()

num_sites = 5
start_health = 100
start_time = 100
prior_threat_level = 0.7
discount_factor = 0.6
threat_seed = 123

settings = SimSettings(num_sites, start_health, start_time, prior_threat_level,
                       discount_factor, threat_seed)

wh = 0.75
reward_model1 = ConstantWeights(wh=wh)
reward_model2 = StateDependentWeights()

robot1 = RobotOnly(reward_model1, settings)
robot2 = RobotOnly(reward_model2, settings)


health = start_health
time = start_time
# for i in range(num_sites):

d_1_star = (1 - wh)/wh

fake_human_info = HumanInfo(100, 100, d_1_star, -1, 0)
wh_sd = reward_model2.get_wh(fake_human_info)
d_2_star = (1 - wh_sd) / wh_sd

# threat_level = settings.threat_setter.after_scan[0]
threat_level = rng.uniform(low=d_1_star, high=d_2_star)
info = RobotInfo(health, time, threat_level, settings.d, 0)
rec1 = robot1.choose_action(info)
rec2 = robot2.choose_action(info)

print(rf"$d_1^*$: {d_1_star}, $d_2^*$: {d_2_star}")
print(f"Threat level {round(threat_level, 2)}, rec1 {rec1}, rec2 {rec2}")
