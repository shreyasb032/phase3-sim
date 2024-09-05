from classes.HumanModels import Human
from classes.RobotModel import Robot
from classes.State import HumanInfo, RobotInfo, Observation
from classes.SimSettings import SimSettings


class Simulation:
    """Class for a single simulation"""

    def __init__(self, settings: SimSettings, robot: Robot, human: Human):
        self.settings = settings
        self.robot = robot
        self.human = human

        self.health_history = [settings.start_health]
        self.time_history = [settings.start_time]
        self.action_history = []
        self.rec_history = []
        self.trust_history = []

    def run(self):
        """
        Runs a simulation of the ISR mission for all sites
        """
        health = self.settings.start_health
        time = self.settings.start_time
        after_scan = self.settings.threat_setter.after_scan
        threats = self.settings.threat_setter.threats
        prior = self.settings.d

        for site_idx in range(self.settings.num_sites):
            threat_level = after_scan[site_idx]
            robot_info = RobotInfo(health, time, threat_level, prior, site_idx)
            rec = self.robot.get_recommendation(robot_info)
            human_info = HumanInfo(health, time, threat_level, rec, site_idx)
            action = self.human.choose_action(human_info)
            threat = threats[site_idx]
            obs = Observation(threat, action)

            # Update the human's trust
            self.human.forward(human_info, obs)

            # Get the trust sample
            trust_fb = self.human.get_trust_sample()

            # Add it to the observation
            obs.add_trust_feedback(trust_fb)

            # Update the robot's model of the human
            self.robot.forward(robot_info, obs)

            # Update the health and time
            health = robot_info.health
            time = robot_info.time

            # Store the data
            self.health_history.append(health)
            self.time_history.append(time)
            self.rec_history.append(rec)
            self.action_history.append(action)
            self.trust_history.append(trust_fb)
