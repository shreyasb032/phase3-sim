from os import path
import pickle
import _context
import numpy as np
from scipy.special import expit
from classes.State import HumanInfo
from classes.RewardModels import StateDependentWeights


def main():
    model_file = path.join('..', 'models', 'model_hc.pkl')
    scaler_file = path.join('..', 'models', 'scaler.pkl')

    with open(model_file, 'rb') as f:
        ols_results = pickle.load(f)

    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    print(ols_results.params)
    print(scaler.mean_)
    # print(scaler.var_)
    print(scaler.scale_)

    # Comparison
    health = 100
    time = 100
    h = health / 100.
    c = time / 100.
    state = np.array([h, c], dtype=float)
    state_normalized = (state - scaler.mean_) / scaler.scale_
    state_with_const = np.insert(state_normalized, 0, [1], axis=0)
    print(state_with_const)
    wh_manual = expit(ols_results.params.dot(state_with_const))
    print(wh_manual)

    rewards_model = StateDependentWeights()
    info = HumanInfo(health, time, 0., 0, 0)
    wh_auto = rewards_model.get_wh(info)
    print(wh_auto)


if __name__ == "__main__":
    main()
