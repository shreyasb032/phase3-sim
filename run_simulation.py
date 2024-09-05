# Goal - compare a non-adaptive strategy with constant weights from the informed prior
#        and one with the learnt state-dependent reward weights (still non-adaptive)

from classes.SimSettings import SimSettings
from classes.Simulation import Simulation
from classes.RobotModel import Robot
from classes.HumanModels import Human, HumanModel
from classes.TrustModels import BetaDistributionModel
from classes.PerformanceMetrics import ObservedReward
from classes.DecisionModels import BoundedRationalityDisuse
from classes.RewardModels import StateDependentWeights
from classes.ParamsGenerator import TrustParamsGenerator
from classes.State import HumanInfo


class SimRunner:
    """
    Sets up and runs the simulation
    """
    def __init__(self):
        self.sim = None
        self.sim_settings = None
        self.robot = None
        self.human = None

    def init_settings(self, num_sites: int = 10, start_health: int = 100, start_time: int = 0,
                      prior_threat_level: float = 0.7, discount_factor: float = 0.7):
        self.sim_settings = SimSettings(num_sites, start_health, start_time, prior_threat_level, discount_factor,
                                        threat_seed=123)

    def init_robot(self):

        # Trust model
        parameters = {"alpha0": 10., "beta0": 10., 'vs': 10., 'vf': 20.}
        performance_metric = ObservedReward()
        trust_model = BetaDistributionModel(parameters, performance_metric, seed=123)

        # Decision model
        decision_model = BoundedRationalityDisuse(kappa=0.2, seed=123)

        # Reward model
        reward_model = StateDependentWeights(add_noise=True)

        # Human model
        human_model = HumanModel(trust_model, decision_model, reward_model)

        # Robot
        self.robot = Robot(human_model, reward_model, self.sim_settings)

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
        self.human = Human(trust_model, decision_model, reward_model)

    def init_sim(self):
        self.init_settings()
        self.init_robot()
        self.init_human()
        self.sim = Simulation(self.sim_settings, self.robot, self.human)

    def run(self):
        self.init_sim()
        self.sim.run()

    def print_results(self):
        print("Site no., Health, Time, Trust, Recommendation, Action, wh")
        print(f"{0:<7}, {100: <7}, {0: <5}")
        trust_history = self.sim.trust_history
        health_history = self.sim.health_history
        time_history = self.sim.time_history
        rec_history = self.sim.rec_history
        action_history = self.sim.action_history
        for i in range(len(self.sim.trust_history)):
            info = HumanInfo(health_history[i], time_history[i], 1.0, rec_history[i], i)
            wh = self.human.reward_model.get_wh(info)
            print(f"{i+1:<7}, {health_history[i+1]:<7}, {time_history[i+1]:<5}, "
                  f"{f'{trust_history[i]:.2f}':<5}, {rec_history[i]:<12}, {action_history[i]:<6}, {wh:.2f}")


def main():
    sim_runner = SimRunner()
    sim_runner.run()
    sim_runner.print_results()


if __name__ == "__main__":
    main()
