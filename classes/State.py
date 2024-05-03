class HumanInfo:
    """Represents the information available to the human"""
    def __init__(self, health: int, time: int, threat_level: float,
                 recommendation: int, site_idx: int):
        self.health = health
        self.time = time
        self.threat_level = threat_level
        self.recommendation = recommendation
        self.site_idx = site_idx


class RobotInfo:
    """Represents the information available to the robot"""
    def __init__(self, health: int, time: int, threat_level: float,
                 threat_level_prior: float, site_idx: int):
        self.health = health
        self.time = time
        self.threat_level = threat_level
        self.prior_threat_level = threat_level_prior
        self.site_idx = site_idx


class Observation:
    """Represents the information gained after observing the outcome"""
    def __init__(self, threat: int):
        self.threat = threat
        self.trust_feedback = None

    def add_trust_feedback(self, trust_feedback):
        self.trust_feedback = trust_feedback


class State:
    """Represents a state"""
    def __init__(self, health, time):
        self.health = health
        self.time = time

        # A unique index given to each state
        self.idx = self.health * 101 + self.time
