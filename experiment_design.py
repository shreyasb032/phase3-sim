import os
import os.path as path
from typing import Dict
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from classes.SimSettings import SimSettings
from classes.Simulation import Simulation
from classes.RewardModels import StateDependentWeights
from classes.State import HumanInfo
from run_simulation import SimRunner

sns.set_theme(context='talk', style='white')

NUM_SITES = 10
PRIOR_THREAT_LEVEL = 0.6
DISCOUNT_FACTOR = 0.7
NUM_PARTICIPANTS_PER_INITIAL = 20
WH_CONST = [0.6, 0.7, 0.8, 0.9]

class ExperimentDesign:
    """
    Class to run multiple simulations with different starting conditions
    and reward weights for the robot
    """

    def __init__(self, starting_conditions):
        self.starting_conditions = starting_conditions
        self.health_bins = np.arange(-50, 110, 10)
        self.time_bins = np.arange(0, 180, 10)

    def run_and_save_sims(self):
        for i, starting_condition in enumerate(self.starting_conditions):
            start_health, start_time = starting_condition
            for j in tqdm(range(NUM_PARTICIPANTS_PER_INITIAL)):
                settings = SimSettings(NUM_SITES, start_health, start_time,
                                       PRIOR_THREAT_LEVEL, DISCOUNT_FACTOR,
                                       threat_seed=10 * i + j)
                sim_runner = SimRunner(settings, wh_const=WH_CONST)
                sim_runner.run()
                file = path.join('data', f'run_{i}_{j}.pkl')
                data = {'sim_runner': sim_runner, 'starting_condition': starting_condition}
                with open(file, 'wb') as f:
                    pickle.dump(data, f)

    def __get_state_counts(self, sim: Simulation, counts: Dict, key: str):
        """
        Goes through the data of the simulation runner and
        returns a list of states visited and their counts to help with
        plotting
        """
        healths = sim.health_history
        times = sim.time_history
        states = np.array([[h, t] for h, t in zip(healths, times)])

        if counts[key] is None:
            counts[key] = np.zeros((self.health_bins.shape[0], self.time_bins.shape[0]),
                              dtype=int)

        for i, health in enumerate(self.health_bins):
            for j, _time in enumerate(self.time_bins):
                count = np.sum(np.all(states == [health, _time], axis=1))
                counts[key][i, j] += count

        return counts

    def __plot_single(self, sim: Simulation, ax: plt.Axes, counts: Dict, key: str):
        """
        Plots the heatmap for a single simulation
        """
        counts = self.__get_state_counts(sim, counts, key)
        im = ax.imshow(counts[key], origin='lower')
        ax.set_yticks(np.arange(len(self.health_bins)), labels=self.health_bins)
        ax.set_xticks(np.arange(len(self.time_bins)), labels=self.time_bins)
        ax.set_title(key)
        ax.set_xlabel('Time spent')
        ax.set_ylabel('Health remaining')
        return ax, counts

    def plot_states_visited(self, dir_path: str | None = None):
        """
        Plots the states visited in the simulation as a heatmap
        """
        if dir_path is None:
            dir_path = './data/'
        files = os.listdir(dir_path)
        axes = {}
        figs = {}
        counts = {}
        for file in files:
            filepath = path.join(dir_path, file)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            sim_runner = data['sim_runner']
            state_dep_sim = sim_runner.state_dep_sim

            # Add a figure and axis for the state dep sim
            if len(axes) == 0:
                fig, ax = plt.subplots()
                axes['state_dep'] = ax
                figs['state_dep'] = fig
                counts['state_dep'] = None

            const_sims = sim_runner.const_sims

            # Add figs and axes for the constant sims
            if len(axes) == 1:
                for sim in const_sims:
                    fig, ax = plt.subplots()
                    info = HumanInfo(100, 100, 1., 1, 0)
                    wh = sim.robot.reward_model.get_wh(info)
                    axes[f'{wh:.2f}'] = ax
                    figs[f'{wh:.2f}'] = fig
                    counts[f'{wh:.2f}'] = None

            # Make a list of all sims
            sims = [state_dep_sim]
            sims.extend(const_sims)

            for sim in sims:
                if isinstance(sim.robot.reward_model, StateDependentWeights):
                    fig = figs['state_dep']
                    ax = axes['state_dep']
                    key = 'state_dep'
                    ax, counts = self.__plot_single(sim, ax, counts, key)
                    fig.tight_layout()
                else:
                    info = HumanInfo(100, 100, 1., 1, 0)
                    wh = sim.robot.reward_model.get_wh(info)
                    key = f'{wh:.2f}'
                    fig = figs[key]
                    ax = axes[key]
                    ax, counts = self.__plot_single(sim, ax, counts, key)
                    fig.tight_layout()

                axes[key] = ax
                figs[key] = fig

        plt.show()

    @staticmethod
    def __trust_data_helper(trust_data: Dict, data: Dict):
        """
        Updates and returns the trust data
        """
        sim_runner = data['sim_runner']
        state_dep_sim = sim_runner.state_dep_sim
        # Add the first entry to trust data
        if len(trust_data) == 0:
            trust_data['state_dep'] = [state_dep_sim.trust_history]
        else:
            trust_data['state_dep'].append(state_dep_sim.trust_history)
        const_sims = sim_runner.const_sims
        # Add the rest
        if len(trust_data) == 1:
            for sim in const_sims:
                info = HumanInfo(100, 100, 1., 1, 0)
                wh = sim.robot.reward_model.get_wh(info)
                key = f'{wh:.2f}'
                trust_data[key] = [sim.trust_history]
        else:
            for sim in const_sims:
                info = HumanInfo(100, 100, 1., 1, 0)
                wh = sim.robot.reward_model.get_wh(info)
                key = f'{wh:.2f}'
                trust_data[key].append(sim.trust_history)

        return trust_data

    @staticmethod
    def __plot_trust_helper(trust_data: Dict, ax: plt.Axes):
        palette = sns.color_palette('deep')
        markers = ['o', 'v', 's', 'P', 'X', '*']
        lw = 2
        alpha = 0.5
        for i, (key, val) in enumerate(trust_data.items()):
            _trust = np.array(val)
            color = palette[i]
            marker = markers[i]
            mean = _trust.mean(axis=0)
            std = _trust.std(axis=0)
            ci = 1.96 * std / np.sqrt(_trust.shape[0])
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, lw=lw, label=key, c=color, marker=marker)
            ax.fill_between(x, mean-ci, mean+ci, color=color, alpha=alpha)

        return ax

    def plot_trust(self, dir_path: str | None = None):
        """
        Plots the trust feedback given for the robot using different strategies
        """
        if dir_path is None:
            dir_path = './data/'
        files = os.listdir(dir_path)
        fig, ax = plt.subplots(figsize=(14, 10))
        trust_data = {}
        for file in files:
            filepath = path.join(dir_path, file)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            trust_data = self.__trust_data_helper(trust_data, data)

        ax = self.__plot_trust_helper(trust_data, ax)
        ax.legend()
        ax.grid('y')
        ax.set_ylim([0.3, 1.0])
        fig.tight_layout()
        plt.show()

def main():
    starting_conditions = [(100, 0), (100, 80), (30, 0), (90, 30), (30, 30), (40, 70), (50, 50), (50, 0),
                           (70, 50),  (20, 70)]
    runner = ExperimentDesign(starting_conditions)
    # runner.run_and_save_sims()
    # runner.plot_states_visited()
    runner.plot_trust()


if __name__ == "__main__":
    main()
