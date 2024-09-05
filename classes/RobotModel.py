import numpy as np
from classes.RewardModels import RewardModelBase
from classes.HumanModels import HumanModel
from classes.SimSettings import SimSettings
from classes.State import RobotInfo, Observation, HumanInfo


class RobotOnly:
    """
    A class for a task where the robot is doing the ISR mission by itself
    """

    def __init__(self, reward_model: RewardModelBase,
                 settings: SimSettings):
        """
        :param reward_model: the rewards model to be used
        :param settings: the simulation settings
        """
        self.rewards_model = reward_model
        self.settings = settings
        self.num_sites = settings.num_sites

    def choose_action(self, info: RobotInfo):
        """
        Chooses an action based on the information available
        :param info: the information available to the robot while choosing an action
        """
        num_sites = self.settings.num_sites
        current_idx = info.site_idx
        number_of_sites_to_go = num_sites - current_idx
        value_matrix = np.zeros((
            number_of_sites_to_go + 1,  # stages
            number_of_sites_to_go + 1,  # possible health
            number_of_sites_to_go + 1  # possible time
        ))

        action_matrix = np.zeros((
            number_of_sites_to_go,
            number_of_sites_to_go,
            number_of_sites_to_go
        ))

        for stage in reversed(range(number_of_sites_to_go)):
            possible_health = info.health - np.arange(stage + 1) * 10
            possible_time = info.time - np.arange(stage + 1) * 10
            for health_idx, health in enumerate(possible_health):
                for time_idx, time in enumerate(possible_time):
                    temp_info = HumanInfo(health, time, info.threat_level, -1, info.site_idx)
                    wh = self.rewards_model.get_wh(temp_info)
                    wc = 1 - wh

                    reward_0 = -wh * info.threat_level  # One-step expected reward for action 0
                    reward_1 = -wc  # One-step expected reward for action 0

                    threat_level = info.prior_threat_level
                    if stage == 0:  # If at the current site, use the updated threat level
                        threat_level = info.threat_level

                    value_0 = (reward_0 +
                               self.settings.df * (threat_level * value_matrix[stage + 1, health_idx + 1, time_idx] +
                                                   (1 - threat_level) * value_matrix[stage + 1, health_idx, time_idx]))
                    value_1 = (reward_1 +
                               self.settings.df * value_matrix[stage + 1, health_idx, time_idx + 1])

                    if value_0 >= value_1:
                        value_matrix[stage, health_idx, time_idx] = value_0
                        action_matrix[stage, health_idx, time_idx] = 0
                    else:
                        value_matrix[stage, health_idx, time_idx] = value_1
                        action_matrix[stage, health_idx, time_idx] = 1

        return action_matrix[0, 0, 0]


class Robot:

    def __init__(self, human_model: HumanModel,
                 reward_model: RewardModelBase,
                 settings: SimSettings):
        self.recommendation = None
        self.human_model = human_model
        self.reward_model = reward_model
        self.settings = settings

    def get_recommendation(self, info: RobotInfo):
        """
        Generates a recommendation from the information the robot has
        :param info: the information available to the robot when making a recommendation
        """
        num_houses_to_go = self.settings.num_sites - info.site_idx
        value_matrix = np.zeros((num_houses_to_go + 1,  # stages
                                 num_houses_to_go + 1,  # success/failure
                                 num_houses_to_go + 1,  # health
                                 num_houses_to_go + 1), dtype=float)  # time
        action_matrix = np.zeros((num_houses_to_go,
                                  num_houses_to_go,
                                  num_houses_to_go,
                                  num_houses_to_go), dtype=int)
        current_health = info.health
        current_time = info.time

        for stage in reversed(range(num_houses_to_go)):
            possible_successes = np.arange(stage + 1)
            possible_failures = stage - possible_successes
            _alpha = self.human_model.trust_model.alpha
            _beta = self.human_model.trust_model.beta
            vs = self.human_model.trust_model.parameters['vs']
            vf = self.human_model.trust_model.parameters['vf']
            possible_healths = current_health - np.arange(stage + 1) * 10
            possible_times = current_time + np.arange(stage + 1) * 10

            df = self.settings.df
            threat_level = info.prior_threat_level
            if stage == 0:
                threat_level = info.threat_level

            for i, (ns, nf) in enumerate(zip(possible_successes, possible_failures)):
                alpha = _alpha + ns * vs
                beta = _alpha + nf * vf
                trust = alpha / (alpha + beta)
                for j, health in enumerate(possible_healths):
                    for k, time in enumerate(possible_times):
                        fake_human_info = HumanInfo(health, time, threat_level, -1, stage)
                        wh = self.reward_model.get_wh(fake_human_info)
                        wc = 1 - wh

                        # Computations for recommending to NOT USE the RARV
                        fake_human_info.recommendation = 0
                        prob_0, prob_1 = self.human_model.decision_model.get_prob_of_actions(fake_human_info, trust,
                                                                                             wh=wh)
                        # immediate expected reward for recommending to not use the RARV
                        reward_0 = -wh * threat_level * prob_0 - wc * prob_1
                        # Future discounted value for recommending to not use the RARV
                        value_0 = (reward_0 +
                                   # Trust gain, no Health loss, no time loss
                                   df * prob_0 * (1 - threat_level) * value_matrix[stage + 1, i, j, k] +
                                   # Trust gain, no Health loss, time loss
                                   df * prob_1 * threat_level * value_matrix[stage + 1, i, j, k + 1] +
                                   # Trust loss, no Health loss, time loss
                                   df * prob_1 * (1 - threat_level) * value_matrix[stage + 1, i + 1, j, k + 1] +
                                   # Trust loss, Health loss, no time loss
                                   df * prob_0 * threat_level * value_matrix[stage + 1, i + 1, j + 1, k])

                        # Computations for recommending to use the RARV
                        fake_human_info.recommendation = 1
                        prob_0, prob_1 = self.human_model.decision_model.get_prob_of_actions(fake_human_info, trust,
                                                                                             wh=wh)
                        reward_1 = -wh * threat_level * prob_0 - wc * prob_1
                        value_1 = (reward_1 +
                                   # Trust gain, no Health loss, no time loss
                                   df * prob_0 * (1 - threat_level) * value_matrix[stage + 1, i, j, k] +
                                   # Trust gain, no Health loss, time loss
                                   df * prob_1 * threat_level * value_matrix[stage + 1, i, j, k + 1] +
                                   # Trust loss, no Health loss, time loss
                                   df * prob_1 * (1 - threat_level) * value_matrix[stage + 1, i + 1, j, k + 1] +
                                   # Trust loss, Health loss, no time loss
                                   df * prob_0 * threat_level * value_matrix[stage + 1, i + 1, j + 1, k])

                        if value_0 > value_1:
                            value_matrix[stage, i, j, k] = value_0
                            action_matrix[stage, i, j, k] = 0
                        else:
                            value_matrix[stage, i, j, k] = value_1
                            action_matrix[stage, i, j, k] = 1

        self.recommendation = action_matrix[0, 0, 0, 0]
        return self.recommendation

    def forward(self, info: RobotInfo, obs: Observation):
        """Updates the robot model after seeing the observations.
        The observations must include the trust feedback received from the human"""
        # Steps: 1. Update human model
        #        2. Update human trust model
        fake_human_info = HumanInfo(info.health, info.time, info.threat_level, self.recommendation, info.site_idx)
        wh = self.reward_model.get_wh(fake_human_info)
        self.human_model.update_trust(fake_human_info, obs, wh)
        self.human_model.update_trust_model(obs.trust_feedback)

        if obs.threat == 1:
            if obs.action_chosen == 0:
                info.health -= 10
        if obs.action_chosen == 1:
            info.time += 10
