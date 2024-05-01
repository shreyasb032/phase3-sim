class HumanInfo:
    """Represents the information available to the human"""
    def __init__(self, health: int, time: int, threat_level: float,
                 recommendation: int):
        self.health = health
        self.time = time
        self.threat_level = threat_level
        self.recommendation = recommendation


class RobotInfo:
    """Represents the information available to the robot"""
    def __init__(self, health: int, time: int, threat_level: float,
                 threat_level_prior: float):
        self.health = health
        self.time = time
        self.threat_level = threat_level
        self.prior_threat_level = threat_level_prior


class Observation:
    """Represents the information gained after observing the outcome"""
    def __init__(self, threat: int):
        self.threat = threat
        self.trust_feedback = None

    def add_trust_feedback(self, trust_feedback):
        self.trust_feedback = trust_feedback
