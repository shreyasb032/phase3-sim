from time import perf_counter
import sys
import os
import os.path as path
from typing import Dict
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from classes.SimSettings import SimSettings
from classes.Simulation import Simulation
from classes.RewardModels import StateDependentWeights
from classes.State import HumanInfo
from run_simulation import SimRunner

sns.set_theme(context='talk', style='white')

NUM_SITES = 10
PRIOR_THREAT_LEVEL = 0.7
DISCOUNT_FACTOR = 0.7
NUM_PARTICIPANTS_PER_INITIAL = 100
# WH_CONST = [0.7, 0.8, 0.87, 0.95]
WH_CONST = [0.8062]


class ExperimentDesign:
    """
    Class to run multiple simulations with different starting conditions
    and reward weights for the robot
    """

    def __init__(self, starting_conditions):
        self.starting_conditions = starting_conditions
        self.health_bins = np.arange(0, 110, 10)
        self.time_bins = np.arange(0, 110, 10)

    def run_and_save_sims(self):
        for i, starting_condition in enumerate(self.starting_conditions):
            start_health, start_time = starting_condition
            for j in tqdm(range(NUM_PARTICIPANTS_PER_INITIAL)):
                settings = SimSettings(NUM_SITES, start_health, start_time,
                                       PRIOR_THREAT_LEVEL, DISCOUNT_FACTOR,
                                       threat_seed=None)
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

    def __plot_single(self, ax: plt.Axes, counts: Dict, key: str):
        """
        Plots the heatmap for a single simulation
        """
        im = ax.imshow(counts[key], origin='lower')
        ax.set_yticks(np.arange(len(self.health_bins)), labels=self.health_bins)
        ax.set_xticks(np.arange(len(self.time_bins)), labels=self.time_bins)
        ax.set_title(key)
        ax.set_xlabel('Time remaining')
        ax.set_ylabel('Health remaining')
        return im, ax, counts

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
            if 'csv' in file:
                continue
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
                    key = 'state_dep'
                    counts = self.__get_state_counts(sim, counts, key)
                else:
                    info = HumanInfo(100, 100, 1., 1, 0)
                    wh = sim.robot.reward_model.get_wh(info)
                    key = f'{wh:.2f}'
                    counts = self.__get_state_counts(sim, counts, key)

        for key, fig in figs.items():
            ax = axes[key]
            im, ax, counts = self.__plot_single(ax, counts, key)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

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
    def __health_data_helper(health_data: Dict, data: Dict):
        """
        Updates and returns the trust data
        """
        sim_runner = data['sim_runner']
        state_dep_sim = sim_runner.state_dep_sim
        # Add the first entry to health data
        if len(health_data) == 0:
            health_data['state_dep'] = [state_dep_sim.health_history]
        else:
            health_data['state_dep'].append(state_dep_sim.health_history)
        const_sims = sim_runner.const_sims

        # Add the rest
        if len(health_data) == 1:
            for sim in const_sims:
                info = HumanInfo(100, 100, 1., 1, 0)
                wh = sim.robot.reward_model.get_wh(info)
                key = f'{wh:.2f}'
                health_data[key] = [sim.health_history]
        else:
            for sim in const_sims:
                info = HumanInfo(100, 100, 1., 1, 0)
                wh = sim.robot.reward_model.get_wh(info)
                key = f'{wh:.2f}'
                health_data[key].append(sim.health_history)

        return health_data

    @staticmethod
    def __plot_health_helper(health_data: Dict, ax: plt.Axes):
        palette = sns.color_palette('deep')
        markers = ['o', 'v', 's', 'P', 'X', '*']
        lw = 2
        alpha = 0.5
        for i, (key, val) in enumerate(health_data.items()):
            _trust = np.array(val)
            color = palette[i]
            marker = markers[i]
            mean = _trust.mean(axis=0)
            std = _trust.std(axis=0)
            ci = 1.96 * std / np.sqrt(_trust.shape[0])
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, lw=lw, label=key, c=color, marker=marker)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=alpha)
            ax.set_xlabel('Interactions')
            ax.set_ylabel('Health')

        return ax

    @staticmethod
    def __time_data_helper(time_data: Dict, data: Dict):
        """
        Updates and returns the trust data
        """
        sim_runner = data['sim_runner']
        state_dep_sim = sim_runner.state_dep_sim
        # Add the first entry to health data
        if len(time_data) == 0:
            time_data['state_dep'] = [state_dep_sim.time_history]
        else:
            time_data['state_dep'].append(state_dep_sim.time_history)
        const_sims = sim_runner.const_sims

        # Add the rest
        if len(time_data) == 1:
            for sim in const_sims:
                info = HumanInfo(100, 100, 1., 1, 0)
                wh = sim.robot.reward_model.get_wh(info)
                key = f'{wh:.2f}'
                time_data[key] = [sim.time_history]
        else:
            for sim in const_sims:
                info = HumanInfo(100, 100, 1., 1, 0)
                wh = sim.robot.reward_model.get_wh(info)
                key = f'{wh:.2f}'
                time_data[key].append(sim.time_history)

        return time_data

    @staticmethod
    def __plot_time_helper(time_data: Dict, ax: plt.Axes):
        palette = sns.color_palette('deep')
        markers = ['o', 'v', 's', 'P', 'X', '*']
        lw = 2
        alpha = 0.5
        for i, (key, val) in enumerate(time_data.items()):
            _trust = np.array(val)
            color = palette[i]
            marker = markers[i]
            mean = _trust.mean(axis=0)
            std = _trust.std(axis=0)
            ci = 1.96 * std / np.sqrt(_trust.shape[0])
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, lw=lw, label=key, c=color, marker=marker)
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=alpha)
            ax.set_xlabel('Interactions')
            ax.set_ylabel('Time')

        return ax

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
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=alpha)
            ax.set_xlabel('Interactions')
            ax.set_ylabel('Trust')

        return ax

    def plot_trust(self, dir_path: str | None = None):
        """
        Plots the trust feedback given for the robot using different strategies
        """
        if dir_path is None:
            dir_path = './data/'
        files = os.listdir(dir_path)
        fig, ax = plt.subplots(figsize=(13, 9))
        trust_data = {}
        for file in files:
            if 'csv' in file:
                continue
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

    def plot_health_and_time(self, dir_path: str | None = None):
        """
        Plots the trust feedback given for the robot using different strategies
        """
        if dir_path is None:
            dir_path = './data/'
        files = os.listdir(dir_path)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 10))
        health_data = {}
        time_data = {}
        for file in files:
            if 'csv' in file:
                continue
            filepath = path.join(dir_path, file)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            health_data = self.__health_data_helper(health_data, data)
            time_data = self.__time_data_helper(time_data, data)

        ax1 = self.__plot_health_helper(health_data, ax1)
        ax1.legend()
        ax1.grid('y')
        ax1.set_ylim([0, 105])

        ax2 = self.__plot_time_helper(time_data, ax2)
        ax2.legend()
        ax2.grid('y')
        ax2.set_ylim([0, 105])

        fig.tight_layout()
        plt.show()

    def plot_trust_separate(self, dir_path: str | None = None):
        """
        Plots the trust dynamics separately for each initial condition
        """
        if dir_path is None:
            dir_path = './data/'
        files = os.listdir(dir_path)
        # Get the indices
        initial_conditions_indices = set()
        participant_indices = set()
        for file in files:
            if 'csv' in file:
                continue
            details = file.strip('.pkl').split('_')
            idx1 = int(details[1])
            idx2 = int(details[2])
            initial_conditions_indices.add(idx1)
            participant_indices.add(idx2)

        trust_data = {}
        label = None
        for i in sorted(initial_conditions_indices):
            trust_data[i] = {}
            for j in sorted(participant_indices):
                filename = f'{dir_path}run_{i}_{j}.pkl'
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                sim_runner = data['sim_runner']
                label = data['starting_condition']
                if 'state_dep' not in trust_data[i]:
                    trust_data[i]['state_dep'] = [sim_runner.state_dep_sim.trust_history]
                else:
                    trust_data[i]['state_dep'].append(sim_runner.state_dep_sim.trust_history)
                if len(trust_data[i]) == 1:
                    for sim in sim_runner.const_sims:
                        info = HumanInfo(100, 100, 1., 1, 0)
                        wh = sim.robot.reward_model.get_wh(info)
                        key = f'{wh:.2f}'
                        trust_data[i][key] = [sim.trust_history]
                else:
                    for sim in sim_runner.const_sims:
                        info = HumanInfo(100, 100, 1., 1, 0)
                        wh = sim.robot.reward_model.get_wh(info)
                        key = f'{wh:.2f}'
                        trust_data[i][key].append(sim.trust_history)

            fig, ax = plt.subplots()
            self.__plot_trust_helper(trust_data[i], ax)
            ax.set_title(f'Starting Health: {label[0]}, Time: {label[1]}')
            ax.legend()
            ax.grid('y')
            ax.set_ylim([0.3, 1.0])
            fig.tight_layout()

        plt.show()

    @staticmethod
    def __initialize_data():
        """
        Initializes data for storage into Excel sheets
        """
        store = {
            'Run ID': [],
            'Participant ID': [],
            'Site Index': [],
            'Health': [],
            'Time': [],
            'Threat Level': [],
            'Threat': [],
            'Recommendation': [],
            'Action': [],
            'Trust': [],
            'wh': []
        }

        return store

    @staticmethod
    def get_wh_history(sim: Simulation):
        wh_history = []
        for i, (h, t) in enumerate(zip(sim.health_history, sim.time_history)):
            info = HumanInfo(h, t, 1.0, 1, i)
            wh = sim.robot.reward_model.get_wh(info)
            wh_history.append(wh)

        return wh_history

    def __store_helper(self, store: Dict, sim: Simulation, i, j):

        data_length = len(sim.health_history)
        store['Run ID'].extend([i] * data_length)
        store['Participant ID'].extend([j] * data_length)
        store['Site Index'].extend(list(np.arange(data_length)))
        store['Health'].extend(sim.health_history)
        store['Time'].extend(sim.time_history)
        threat_levels = [np.nan]
        threat_levels.extend(sim.threat_level_history)
        threats = [np.nan]
        threats.extend(sim.threat_history)
        recommendations = [np.nan]
        recommendations.extend(sim.rec_history)
        actions = [np.nan]
        actions.extend(sim.action_history)
        trust = [np.nan]
        trust.extend(sim.trust_history)
        store['Threat Level'].extend(threat_levels)
        store['Threat'].extend(threats)
        store['Recommendation'].extend(recommendations)
        store['Action'].extend(actions)
        store['Trust'].extend(trust)
        store['wh'].extend(self.get_wh_history(sim))

        return store

    def convert_to_excel(self, dir_path: str | None = None):
        """
        Converts the saved data to Excel
        One sheet per reward weight in the simulation
        sheet 1 - state_dep
        sheet 2... - 'wh:.2f'
        """
        # Columns - Run ID, Participant ID, Site Index, Health, Time, Threat Level, Threat, Recommendation, Action,
        #           Trust, wh
        if dir_path is None:
            dir_path = './data/'
        files = os.listdir(dir_path)
        # Get the indices
        initial_conditions_indices = set()
        participant_indices = set()
        for file in files:
            if 'csv' in file:
                continue
            details = file.strip('.pkl').split('_')
            idx1 = int(details[1])
            idx2 = int(details[2])
            initial_conditions_indices.add(idx1)
            participant_indices.add(idx2)

        state_dep_store = {}
        const_stores = []
        wh_consts = []
        for i in sorted(initial_conditions_indices):
            for j in sorted(participant_indices):
                filename = f'{dir_path}run_{i}_{j}.pkl'
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                sim_runner = data['sim_runner']
                state_dep_sim = sim_runner.state_dep_sim
                if i == 0 and j == 0:
                    state_dep_store = self.__initialize_data()
                state_dep_store = self.__store_helper(state_dep_store, state_dep_sim, i, j)

                const_sims = sim_runner.const_sims
                for k, sim in enumerate(const_sims):
                    if i == 0 and j == 0:
                        const_stores.append(self.__initialize_data())
                        wh_consts = [None for _ in range(len(const_sims))]
                    const_stores[k] = self.__store_helper(const_stores[k], sim, i, j)
                    wh_consts[k] = const_stores[k]['wh'][0]

        # Convert to pandas dataframe and save to csv
        with pd.ExcelWriter('data/csv/sims.xlsx') as writer:
            df = pd.DataFrame(state_dep_store)
            df.to_excel(writer, sheet_name='state_dep', index=False)
            for i, const_store in enumerate(const_stores):
                df = pd.DataFrame(const_store)
                df.to_excel(writer, sheet_name=f'{wh_consts[i]:.2f}', index=False)


def main():
    # Health remaining and time remaining
    starting_conditions = [(100, 100), (100, 70), (100, 40),
                           (70, 100), (70, 70), (70, 40),
                           (40, 100), (40, 70), (40, 40)]
    runner = ExperimentDesign(starting_conditions)
    runner.run_and_save_sims()
    # runner.plot_states_visited()
    # runner.plot_trust()
    # runner.plot_health_and_time()
    # runner.plot_trust_separate()
    # runner.convert_to_excel()


if __name__ == "__main__":
    main()
