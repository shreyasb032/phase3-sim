import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from classes.SimSettings import SimSettings
from run_simulation import SimRunner

NUM_SITES = 10
PRIOR_THREAT_LEVEL = 0.6
DISCOUNT_FACTOR = 0.7
NUM_PARTICIPANTS_PER_INITIAL = 20


def main():
    starting_conditions = [(100, 0), (100, 80), (30, 0), (90, 30), (30, 30), (40, 70), (50, 50), (50, 0),
                           (70, 50),  (20, 70)]
    states_1 = []
    states_2 = []

    state_dep_trust = {}
    const_trust = {}

    for i, starting_condition in enumerate(starting_conditions):
        start_health, start_time = starting_condition
        trust_1 = []
        trust_2 = []
        for j in tqdm(range(NUM_PARTICIPANTS_PER_INITIAL)):
            settings = SimSettings(NUM_SITES, start_health, start_time,
                                   PRIOR_THREAT_LEVEL, DISCOUNT_FACTOR,
                                   threat_seed=10*i+j)
            sim_runner = SimRunner(settings)
            sim_runner.run()
            state_dep_sim = sim_runner.state_dep_sim
            trust_1.append(state_dep_sim.trust_history)
            for h, t in zip(state_dep_sim.health_history, state_dep_sim.time_history):
                states_1.append([h, t])

            constant_sim = sim_runner.const_sim
            trust_2.append(constant_sim.trust_history)
            for h, t in zip(constant_sim.health_history, constant_sim.time_history):
                states_2.append([h, t])
        state_dep_trust[starting_condition] = trust_1
        const_trust[starting_condition] = trust_2

    # Save the trust data
    with open('state_dep_trust.pkl', 'wb') as f:
        pickle.dump(state_dep_trust, f)

    with open('const_trust.pkl', 'wb') as f:
        pickle.dump(const_trust, f)

    print("State dependent:")
    states_1 = np.array(states_1)

    # Get the counts
    min_h, min_t = states_1.min(axis=0)
    max_h, max_t = states_1.max(axis=0)
    healths = np.arange(min_h, max_h + 10, 10)
    times = np.arange(min_t, max_t + 10, 10)
    counts = np.zeros((healths.shape[0], times.shape[0]), dtype=int)

    for i, health in enumerate(healths):
        for j, time in enumerate(times):
            count = np.sum(np.all(states_1 == [health, time], axis=1))
            counts[i, j] = count

    data = {"Health": healths, "Time": times, "Counts": counts, "States": states_1}
    with open('state_dep_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    fig, ax = plt.subplots()
    im = ax.imshow(counts, origin='lower')
    ax.set_yticks(np.arange(len(healths)), labels=healths)
    ax.set_xticks(np.arange(len(times)), labels=times)
    fig.tight_layout()

    print("\n\nConstant")
    states_2 = np.array(states_2)
    # Get the counts
    min_h, min_t = states_2.min(axis=0)
    max_h, max_t = states_2.max(axis=0)
    healths = np.arange(min_h, max_h + 10, 10)
    times = np.arange(min_t, max_t + 10, 10)

    counts = np.zeros((healths.shape[0], times.shape[0]), dtype=int)

    for i, health in enumerate(healths):
        for j, time in enumerate(times):
            count = np.sum(np.all(states_2 == [health, time], axis=1))
            counts[i, j] = count

    data = {"Health": healths, "Time": times, "Counts": counts, "States": states_1}
    with open('const_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open('starting_conditions.pkl', 'wb') as f:
        pickle.dump(starting_conditions, f)

    fig, ax = plt.subplots()
    im = ax.imshow(counts, origin='lower')
    ax.set_yticks(np.arange(len(healths)), labels=healths)
    ax.set_xticks(np.arange(len(times)), labels=times)
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
