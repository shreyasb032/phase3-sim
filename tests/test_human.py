import _context
from classes.ParamsGenerator import TrustParamsGenerator
from classes.HumanModels import Human
from classes.TrustModels import BetaDistributionModel
from classes.PerformanceMetrics import ObservedReward
from classes.DecisionModels import BoundedRationalityDisuse
from classes.RewardModels import ConstantWeights, StateDependentWeights
from numpy.random import default_rng
from classes.State import HumanInfo, Observation


class HumanTester:
    def __init__(self, use_constant: bool = False):
        self.params_gen = TrustParamsGenerator()
        self.trust_params = self.params_gen.generate()
        self.performance_metric = ObservedReward()
        self.trust_model = BetaDistributionModel(self.trust_params, self.performance_metric)
        self.decision_model = BoundedRationalityDisuse(kappa=0.2)
        self.reward_model = ConstantWeights(wh=0.81)
        if not use_constant:
            self.reward_model = StateDependentWeights()
        self.human = Human(self.trust_model, self.decision_model, self.reward_model)

    def test_decision_model(self):
        info = HumanInfo(health=100, time=100, threat_level=1.0, recommendation=1, site_idx=0)
        trust = 0.8
        for i in range(10):
            act = self.decision_model.choose_action(info, trust, wh=0.1)
            print(act)

    def test_trust_update(self):
        pass

    def test_performance_metric(self):
        healths = [20 * i for i in range(1, 6)]
        times = healths.copy()
        rng = default_rng()
        for h in healths:
            for c in times:
                threat_level = 0.8
                threat = rng.integers(0, 2)
                info1 = HumanInfo(h, c, threat_level, recommendation=0, site_idx=0)
                info2 = HumanInfo(h, c, threat_level, recommendation=1, site_idx=0)
                wh = self.reward_model.get_wh(info1)
                obs = Observation(threat, 0)
                perf1 = self.performance_metric.get_performance(info1, obs, wh)
                perf2 = self.performance_metric.get_performance(info2, obs, wh)
                print(f"wh: {wh:.2f}, Threat: {threat}, Performance 0: {perf1}, Performance 1: {perf2}")

    def test_params_generator(self):
        for i in range(10):
            trust_params = self.params_gen.generate()
            print(f"Iteration: {i}   Alpha0: {trust_params[0]:.2f}, Beta0: {trust_params[1]:.2f}, "
                  f"ws: {trust_params[2]:.2f}, wf: {trust_params[3]:.2f}")

    def test_reward_model(self):
        healths = [20 * i for i in range(1, 6)]
        times = healths.copy()
        for h in healths:
            for c in times:
                info = HumanInfo(h, c, threat_level=0.5, recommendation=0, site_idx=0)
                wh = self.reward_model.get_wh(info)
                print(f"Health: {h}, Time: {c}, wh: {wh: .2f}")


def main():
    tester = HumanTester(use_constant=False)
    # tester.test_params_generator()
    # tester.test_reward_model()
    # tester.test_performance_metric()
    tester.test_decision_model()


if __name__ == "__main__":
    main()
