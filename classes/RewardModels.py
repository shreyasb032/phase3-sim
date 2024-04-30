import pickle
import os
import pandas as pd
import numpy as np
from scipy.special import expit


class RewardModelBase:
    """
    A base class for all reward models
    """

    def __init__(self):
        pass

    def get_wh(self, **kwargs) -> float:
        """
        Inheriting classes should implement this function to returns the health reward weight
        :param kwargs: additional keyword arguments associated with specific reward models
        """
        raise NotImplementedError


class ConstantWeights(RewardModelBase):

    def __init__(self, wh: float):
        """
        :param wh: The health reward weight (fixed throughout interaction)
        """
        super().__init__()
        self.wh = wh

    def get_wh(self, **kwargs) -> float:
        """
        Inheriting classes should implement this function to return a tuple -> (health-loss-reward, time-loss-reward)
        :param kwargs: additional keyword arguments associated with specific reward models
        """

        return self.wh


class StateDependentWeights(RewardModelBase):

    def __init__(self, model_path: str | None = None, scaler_path: str | None = None, add_noise: bool = False):
        """
        :param model_path: The path to a pickle saved statsmodels OLSResults object (default: None)
        :param scaler_path: The path to a pickle saved scipy StandardScaler object (default: None)
        """
        super().__init__()

        # load the model
        self.model_path = model_path
        if model_path is None:
            self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_hc.pkl'))
        with open(self.model_path, 'rb') as f:
            self.ols_results = pickle.load(f)

        # load the scaler
        self.scaler_path = scaler_path
        if scaler_path is None:
            self.scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl'))
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.add_noise = add_noise
        self.rng = None
        if self.add_noise:
            self.rng = np.random.default_rng(seed=123)

    def get_wh(self, **kwargs) -> float:
        """
        :param kwargs: additional keyword arguments associated with specific reward models
                        should contain health, time
        :return:
        """
        health = kwargs['health'] / 100.
        time = kwargs['time'] / 100.
        x_arr = np.array([health, time], dtype=float).reshape((1, 2))
        x_pd = pd.DataFrame(data=x_arr, columns=['h', 'c'])
        x_scaled = self.scaler.transform(x_pd)
        # x_scaled_with_constant = sm.add_constant(x_scaled)
        x_scaled_with_constant = np.insert(x_scaled, 0, 1., axis=1)
        y = x_scaled_with_constant @ self.ols_results.params
        wh = expit(y)

        if self.add_noise:
            wh += self.rng.normal(loc=0.0, scale=0.05)
            wh = max(0.501, wh)

        return wh
