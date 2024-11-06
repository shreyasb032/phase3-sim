from classes.RewardModels import RewardModelBase
from classes.TrustModels import BetaDistributionModel
from classes.DecisionModels import BoundedRationalityDisuse
from classes.ParamsUpdater import Estimator
from classes.State import HumanInfo, Observation


class Human:
    """
    This is the class for the simulated human. It has a fixed trust_model, decision_model, and reward_model
    """

    def __init__(self, trust_model: BetaDistributionModel, decision_model: BoundedRationalityDisuse,
                 reward_model: RewardModelBase):
        """
        :param trust_model: the trust dynamics model of this human
        :param decision_model: the decision-making model of this human
        :param reward_model: the reward weights model of this human
        """
        self.trust_model = trust_model
        self.decision_model = decision_model
        self.reward_model = reward_model

    def forward(self, info: HumanInfo, obs: Observation):
        """
        Updates the human after seeing the observation
        """
        self.update_trust(info, obs, self.reward_model.get_wh(info))

    def update_trust(self, info: HumanInfo, obs: Observation, wh: float):
        """Update trust based on immediate observed reward
        :param info: the information available to the human at the time of decision-making
        :param obs: the observation of the outcome of action selection
        :param wh: the health reward weight of the human
        """
        self.trust_model.update_trust(info, obs, wh)

    def get_trust_mean(self):
        """Returns the mean level of trust"""
        return self.trust_model.trust_mean

    def get_trust_sample(self):
        """Samples trust from the beta distribution"""
        return self.trust_model.trust_sampled

    def choose_action(self, info: HumanInfo) -> int:
        """
        Chooses an action
        :param info: the information available to the human at the time of decision-making
        :return: the chosen action 0 or 1
        """
        trust = self.trust_model.trust_sampled
        wh = self.reward_model.get_wh(info)
        return self.decision_model.choose_action(info, trust, wh=wh)


class HumanModel(Human):
    """
    The human model maintained by the robot. This one will have a variable trust dynamics model which is updated
    after getting trust feedback from the simulated human
    """
    def __init__(self, trust_model: BetaDistributionModel, decision_model: BoundedRationalityDisuse,
                 reward_model: RewardModelBase):
        super().__init__(trust_model, decision_model, reward_model)
        self.trust_model_updater = Estimator()

    def update_trust_model(self, trust_feedback: float, performance: int):
        self.trust_model.parameters = self.trust_model_updater.update_model(trust_feedback, performance)
