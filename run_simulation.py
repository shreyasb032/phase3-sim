# Goal - compare a non-adaptive strategy with constant weights from the informed prior
#        and one with the learnt state-dependent reward weights (still non-adaptive)
from copy import deepcopy
import numpy as np
import pandas as pd
from classes.SimSettings import SimSettings
from classes.Simulation import Simulation
from classes.RobotModel import Robot
from classes.HumanModels import Human, HumanModel
from classes.TrustModels import BetaDistributionModel
from classes.PerformanceMetrics import ObservedReward
from classes.DecisionModels import BoundedRationalityDisuse
from classes.RewardModels import StateDependentWeights, ConstantWeights
from classes.ParamsGenerator import TrustParamsGenerator
from classes.State import HumanInfo


class SimRunner:
    """
    Sets up and runs the simulation
    """
    def __init__(self, settings: SimSettings, wh_const: float = 0.85):
        self.state_dep_sim = None
        self.const_sim = None
        self.sim_settings = settings
        self.state_dep_robot = None
        self.const_robot = None
        self.wh_const = wh_const

        # Two human instances with shared initial parameters
        # This is to ensure that one model only gets updated with recommendations from one robot
        self.human1 = None
        self.human2 = None

    def init_robots(self):

        # Trust model
        parameters = {"alpha0": 10., "beta0": 10., 'vs': 10., 'vf': 20.}
        performance_metric = ObservedReward()
        trust_model = BetaDistributionModel(parameters, performance_metric, seed=123)

        # Decision model
        decision_model = BoundedRationalityDisuse(kappa=0.2, seed=123)

        # Reward model
        reward_model = StateDependentWeights(add_noise=False)

        # Human model
        human_model = HumanModel(trust_model, decision_model, reward_model)

        # Robot with state dependent reward weights
        self.state_dep_robot = Robot(human_model, reward_model, self.sim_settings)

        # Trust model
        parameters = {"alpha0": 10., "beta0": 10., 'vs': 10., 'vf': 20.}
        performance_metric = ObservedReward()
        trust_model = BetaDistributionModel(parameters, performance_metric, seed=123)

        # Decision model
        decision_model = BoundedRationalityDisuse(kappa=0.2, seed=123)

        # Reward model
        reward_model = ConstantWeights(wh=self.wh_const)

        # Human model
        human_model = HumanModel(trust_model, decision_model, reward_model)

        # Robot
        self.const_robot = Robot(human_model, reward_model, self.sim_settings)

    def init_human(self):

        params_generator = TrustParamsGenerator(seed=123, add_noise=True)
        params_list = params_generator.generate()
        parameters = dict(zip(['alpha0', 'beta0', 'vs', 'vf'], params_list))
        performance_metric = ObservedReward()
        trust_model = BetaDistributionModel(parameters, performance_metric, seed=123)

        # Decision model
        decision_model = BoundedRationalityDisuse(kappa=0.2, seed=123)

        # Reward model
        reward_model = StateDependentWeights(add_noise=False)

        # Human
        self.human1 = Human(trust_model, decision_model, reward_model)
        self.human2 = Human(trust_model, decision_model, reward_model)

    def init_sim(self):
        self.init_robots()
        self.init_human()
        self.state_dep_sim = Simulation(self.sim_settings, self.state_dep_robot, self.human1)
        self.const_sim = Simulation(self.sim_settings, self.const_robot, self.human2)

    def run(self):
        self.init_sim()
        self.state_dep_sim.run()
        self.const_sim.run()

    def __print_helper(self, sim):
        health_history = sim.health_history
        time_history = sim.time_history
        rec_history = sim.rec_history
        wh_list = []
        for i in range(len(sim.trust_history)):
            info = HumanInfo(health_history[i], time_history[i], 1.0, rec_history[i], i)
            wh = self.human1.reward_model.get_wh(info)
            wh_list.append(wh)

        # Things to print: Site index, Threat, Threat Level, Health, Time, Recommendation, Trust, Action, wh
        data = {'Site no.': list(np.arange(self.sim_settings.num_sites)),
                'Threat': self.state_dep_sim.threat_history,
                'Threat level': self.state_dep_sim.threat_level_history, 'Health': sim.health_history[1:],
                'Time': sim.time_history[1:], 'Recommendation': sim.rec_history,
                'Trust': sim.trust_history, 'Action': sim.action_history,
                'wh': wh_list}

        df = pd.DataFrame(data)
        print(df)

    def print_results(self):
        print('Robot using state dependent rewards')
        self.__print_helper(self.state_dep_sim)

        print('\n\nRobot using constant rewards')
        self.__print_helper(self.const_sim)


def main():
    num_sites = 10
    start_health = 100
    start_time = 0
    prior_threat_level = 0.7
    discount_factor = 0.8

    settings = SimSettings(num_sites, start_health, start_time, prior_threat_level, discount_factor,
                           threat_seed=123)
    sim_runner = SimRunner(settings)
    sim_runner.run()
    sim_runner.print_results()


if __name__ == "__main__":
    main()
