from classes.State import HumanInfo, Observation


class PerformanceMetricBase:

    def __init__(self):
        """
        Base class. Other classes should inherit from this class and implement the methods below.
        """
        pass

    def get_performance(self, info: HumanInfo, obs: Observation, wh: float):
        """
        Computes the performance of the recommendation. Classes inheriting from this base class should implement this
        function
        :param info: the information available to the human at the time of decision-making
        :param obs: the observation of the outcome of action selection
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

    def get_performance(self, info: HumanInfo, obs: Observation, wh: float):
        """
        Computes the performance of the recommendation. Classes inheriting from this base class should implement this
        function
        :param info: the information available to the human at the time of decision-making
        :param obs: the observation of the outcome of action selection
        :param wh: The health reward weight of the human
        :return: an integer representing the performance of the recommendation
        """
        wc = 1 - wh
        reward_for_recommended_action = -wh * obs.threat * (1 - info.recommendation) - wc * info.recommendation
        reward_for_other_action = -wh * obs.threat * info.recommendation - wc * (1 - info.recommendation)

        return int(reward_for_recommended_action >= reward_for_other_action)


class ImmediateExpectedReward(PerformanceMetricBase):

    def __init__(self):
        """
        The metric that gives a performance value after observing rewards from the environment
        """
        super().__init__()

    def get_performance(self, info: HumanInfo, obs: Observation, wh: float):
        """
        Computes the performance of the recommendation. Classes inheriting from this base class should implement this
        function
        :param info: the information available to the human at the time of decision-making
        :param obs: the observation of the outcome of action selection
        :param wh: The health reward weight of the human
        :return: an integer representing the performance of the recommendation
        """
        wc = 1 - wh
        reward_for_recommended_action = -wh * info.threat_level * (1 - info.recommendation) - wc * info.recommendation
        reward_for_other_action = -wh * info.threat_level * info.recommendation - wc * (1 - info.recommendation)

        return int(reward_for_recommended_action >= reward_for_other_action)
