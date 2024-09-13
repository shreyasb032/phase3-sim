from classes.ThreatSetter import ThreatSetter


class SimSettings:
    """Class for storing simulation settings"""

    def __init__(self, num_sites: int, start_health: int, start_time: int,
                 prior_threat_level: float,
                 discount_factor: float, threat_seed: int | None = None):
        self.num_sites = num_sites
        self.start_health = start_health
        self.start_time = start_time
        self.d = prior_threat_level
        self.df = discount_factor
        self.threat_setter = ThreatSetter(num_sites, prior_threat_level, threat_seed)
        self.threat_setter.set_threats()

    def update_threats(self, prior_threat_level: float, threat_seed: int = 123):
        self.threat_setter = ThreatSetter(self.num_sites, prior_threat_level, threat_seed)
        self.threat_setter.set_threats()
