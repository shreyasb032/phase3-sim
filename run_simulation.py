# Goal - compare a non-adaptive strategy with constant weights from the informed prior
#        and one with the learnt state-dependent reward weights (still non-adaptive)
from time import perf_counter
import sys
from typing import List
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
    def __init__(self, settings: SimSettings, wh_const: List[float]):
        self.state_dep_sim = None
        self.const_sims = None
        self.sim_settings = settings
        self.state_dep_robot = None
        self.const_robots = None
        self.wh_const = wh_const

        # N + 1 human instances with shared initial parameters
        # This is to ensure that one model only gets updated with recommendations from one robot
        self.state_dep_human = None
        self.const_humans = None

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
        self.const_robots = []

        for wh in self.wh_const:
            # Trust model
            parameters = {"alpha0": 10., "beta0": 10., 'vs': 10., 'vf': 20.}
            performance_metric = ObservedReward()
            trust_model = BetaDistributionModel(parameters, performance_metric, seed=123)

            # Decision model
            decision_model = BoundedRationalityDisuse(kappa=0.2, seed=123)

            # Reward model
            reward_model = ConstantWeights(wh=wh)

            # Human model
            human_model = HumanModel(trust_model, decision_model, reward_model)

            # Robot
            self.const_robots.append(Robot(human_model, reward_model, self.sim_settings))

    def init_humans(self):

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
        self.state_dep_human = Human(trust_model, decision_model, reward_model)
        self.const_humans = []

        for _ in range(len(self.wh_const)):
            self.const_humans.append(Human(trust_model, decision_model, reward_model))

    def init_sim(self):
        self.init_robots()
        self.init_humans()
        self.state_dep_sim = Simulation(self.sim_settings, self.state_dep_robot, self.state_dep_human,
                                        choose_smartly=True)
        self.const_sims = []
        for i in range(len(self.wh_const)):
            self.const_sims.append(Simulation(self.sim_settings, self.const_robots[i], self.const_humans[i],
                                              choose_smartly=False))

    def run(self):
        # start = perf_counter()
        self.init_sim()
        # end = perf_counter()
        # print(f"Time for initializing the simulation {end - start: .4f}")
        # start = end
        # The below takes about 10 seconds
        self.state_dep_sim.run()
        # end = perf_counter()
        # print(f"Time for running the state-dependent simulation {end - start: .4f}")
        # start = end
        # One constant sim takes about 4 seconds
        for const_sim in self.const_sims:
            settings = const_sim.settings
            settings.threat_setter.after_scan = np.array(self.state_dep_sim.threat_level_history)
            settings.threat_setter.threats = np.array(self.state_dep_sim.threat_history)
            const_sim.update_settings(settings)
            const_sim.run()
        # end = perf_counter()
        # print(f"Time for running the constant simulation {end - start: .4f}")
        # sys.exit(1)

    def __print_helper(self, sim):
        health_history = sim.health_history
        time_history = sim.time_history
        rec_history = sim.rec_history
        wh_list = []
        for i in range(len(sim.trust_history)):
            info = HumanInfo(health_history[i], time_history[i], 1.0, rec_history[i], i)
            wh = self.state_dep_human.reward_model.get_wh(info)
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

        for i, wh in enumerate(self.wh_const):
            print(f'\n\nRobot using constant rewards {wh:.2f}')
            self.__print_helper(self.const_sims[i])


def main():
    num_sites = 10
    start_health = 100
    start_time = 0
    prior_threat_level = 0.7
    discount_factor = 0.8
    wh_const = [0.6, 0.7, 0.8, 0.9]

    settings = SimSettings(num_sites, start_health, start_time, prior_threat_level, discount_factor,
                           threat_seed=123)
    sim_runner = SimRunner(settings, wh_const)
    sim_runner.run()
    sim_runner.print_results()


if __name__ == "__main__":
    main()
