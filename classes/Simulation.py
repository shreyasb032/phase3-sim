from classes.ThreatSetter import ThreatSetter
from classes.POMDPSolver import Solver
from classes.HumanModels import Human


class SimSettings:
    """Class for storing simulation settings"""

    def __init__(self, num_sites: int, start_health: int, start_time: int,
                 prior_threat_level: float, threat_seed: int = 123):
        self.num_sites = num_sites
        self.start_health = start_health
        self.start_time = start_time
        self.d = prior_threat_level
        self.threat_setter = ThreatSetter(num_sites, prior_threat_level, threat_seed)
        self.threat_setter.set_threats()


class Simulation:
    """Class for a single simulation"""

    def __init__(self, settings: SimSettings, robot: Solver, human: Human):
        self.settings = settings
        self.robot = robot
        self.human = human

    def run(self):
        pass
