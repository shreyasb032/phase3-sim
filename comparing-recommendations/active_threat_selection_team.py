"""Present a threat level between d_1 and d_2 and check if the recommendations from the fixed-weights and
state-dependent weights robots are different"""

import _context
from layout import app
from classes.State import RobotInfo, HumanInfo
from classes.Simulation import SimSettings
from classes.RewardModels import ConstantWeights, StateDependentWeights
from classes.RobotModel import Robot
from classes.HumanModels import Human, HumanModel
from classes.PerformanceMetrics import ObservedReward
from classes.TrustModels import BetaDistributionModel
from classes.DecisionModels import BoundedRationalityDisuse
from dash import Input, Output
from copy import deepcopy
from numpy.random import default_rng


rng = default_rng(seed=123)
num_sites = 5
start_health = 100
start_time = 100
prior_threat_level = 0.7
discount_factor = 0.6
threat_seed = 123

settings = SimSettings(num_sites, start_health, start_time, prior_threat_level,
                       discount_factor, threat_seed)

wh = 0.87
const_reward_model = ConstantWeights(wh=wh)
state_dep_reward_model = StateDependentWeights()

parameters = {'alpha0': 10., 'beta0': 50., 'vs': 10., 'vf': 20.}
performance_metric = ObservedReward()
trust_model = BetaDistributionModel(parameters, performance_metric, seed=rng.integers(low=0, high=20))
decision_model = BoundedRationalityDisuse(kappa=0.1, seed=rng.integers(low=0, high=20))
human = Human(trust_model, decision_model, state_dep_reward_model)

trust_model_robot = deepcopy(trust_model)
for k, v in trust_model_robot.parameters.items():
    trust_model_robot.parameters[k] = v + 5

human_model = HumanModel(trust_model_robot, decision_model, state_dep_reward_model)
robot1 = Robot(human_model, const_reward_model, settings)
robot2 = Robot(human_model, state_dep_reward_model, settings)

const_robot = Robot(human_model, const_reward_model, settings)
state_dep_robot = Robot(human_model, state_dep_reward_model, settings)


# Function to update the recommendations after changing the value of d_hat
@app.callback(
    Output(component_id='constant-rec', component_property='children'),
    Input(component_id='d-hat-input', component_property='value'),
    Input(component_id='threat-level-slider', component_property='value'),
    Input(component_id='wh-slider', component_property='value')
)
def update_const_recommendation(d_hat, d, _wh):
    settings.update_threats(d)
    const_robot.settings = settings
    health = settings.start_health
    time = settings.start_time
    info = RobotInfo(health, time, d_hat, d, 0)
    const_reward_model_new = ConstantWeights(wh=_wh)
    const_robot.reward_model = const_reward_model_new
    action = const_robot.get_recommendation(info)
    return f"constant recommendation: {action}"


@app.callback(
    Output(component_id='state-rec', component_property='children'),
    Input(component_id='d-hat-input', component_property='value'),
    Input(component_id='threat-level-slider', component_property='value'),
    Input(component_id='health-slider', component_property='value'),
    Input(component_id='time-slider', component_property='value')
)
def update_state_dep_recommendation(d_hat, d, h, c):
    settings.update_threats(d)
    settings.start_time = c
    settings.start_health = h
    state_dep_robot.settings = settings
    info = RobotInfo(h, c, d_hat, d, 0)
    action = state_dep_robot.get_recommendation(info)
    return f"state dependent recommendation: {action}"


if __name__ == "__main__":
    app.run(debug=True)
