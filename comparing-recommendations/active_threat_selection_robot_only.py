"""Present a threat level between d_1 and d_2 and check if the recommendations from the fixed-weights and
state-dependent weights robots are different"""

import _context
from layout import app
from classes.State import RobotInfo
from classes.Simulation import SimSettings
from classes.RewardModels import ConstantWeights, StateDependentWeights
from classes.RobotModel import RobotOnly
from dash import Input, Output

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
const_robot = RobotOnly(const_reward_model, settings)

state_dep_reward_model = StateDependentWeights()
state_dep_robot = RobotOnly(state_dep_reward_model, settings)


# Function to update the recommendations after changing the value of d_hat
@app.callback(
    Output(component_id='constant-rec', component_property='children'),
    Input(component_id='d-hat-input', component_property='value'),
    Input(component_id='threat-level-slider', component_property='value')
)
def update_const_recommendation(d_hat, d):
    settings.update_threats(d)
    const_robot.settings = settings
    health = settings.start_health
    time = settings.start_time
    info = RobotInfo(health, time, d_hat, d, 0)
    action = const_robot.choose_action(info)
    return f"constant optimal action: {action}"


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
    action = state_dep_robot.choose_action(info)
    return f"state dependent optimal action: {action}"


if __name__ == "__main__":
    app.run(debug=True)
