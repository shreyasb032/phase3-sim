from typing import Dict
from classes.RewardModels import RewardModelBase


class PerformanceMetricBase:

    def __init__(self):
        """
        Base class. Other classes should inherit from this class and implement the methods below.
        """
        pass

    def get_performance(self, recommendation: int, threat: int, threat_level: float, wh: float):
        """
        Computes the performance of the recommendation. Classes inheriting from this base class should implement this
        function
        :param recommendation: The recommended action by the system
        :param threat: The observed presence of threat
        :param threat_level: The threat level indicated by the drone
        :param wh: The health reward weight of the human
        :return: an integer representing the performance of the recommendation
        """
        raise NotImplementedError


class ObservedReward(PerformanceMetricBase):

    def __init__(self):
        """
        The metric that gives a performance value after observing rewards from the environment
        """
        super().__init__()

    def get_performance(self, recommendation: int, threat: int, threat_level: float, wh: float):
        """
        Computes the performance of the recommendation by comparing the observed rewards for the two actions
        :param recommendation: The recommended action by the system
        :param threat: The observed presence of threat
        :param threat_level: The threat level indicated by the drone
        :param wh: The health reward weight of the human
        :return: an integer representing the performance of the recommendation
        """
        wc = 1 - wh
        reward_for_recommended_action = -wh * threat * (1 - recommendation) - wc * recommendation
        reward_for_other_action = -wh * threat * recommendation - wc * (1 - recommendation)

        return int(reward_for_recommended_action >= reward_for_other_action)


class ImmediateExpectedReward(PerformanceMetricBase):

    def __init__(self):
        """
        The metric that gives a performance value after observing rewards from the environment
        """
        super().__init__()

    def get_performance(self, recommendation: int, threat: int, threat_level: float, wh: float):
        """
        Computes the performance of the recommendation by comparing the observed rewards for the two actions
        :param recommendation: The recommended action by the system
        :param threat: The observed presence of threat
        :param threat_level: The threat level indicated by the drone
        :param wh: The health reward weight of the human
        :return: an integer representing the performance of the recommendation
        """
        wc = 1 - wh
        reward_for_recommended_action = -wh * threat_level * (1 - recommendation) - wc * recommendation
        reward_for_other_action = -wh * threat_level * recommendation - wc * (1 - recommendation)

        return int(reward_for_recommended_action >= reward_for_other_action)
