import numpy as np
import json
import os.path as path

prob_bdm = 31. / 45.
prob_disbeliever = 5. / 45.
prob_oscillator = 1.0 - prob_bdm - prob_disbeliever


class TrustParamsGenerator:
    """
    Uses the data from phase 1 of the study to sample trust parameters
    """
    def __init__(self, seed=123, add_noise=False):
        self.params_list = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.add_noise = add_noise
        self.params = {}
        self.group = None

        self.bdm_dict = None
        self.oscillator_dict = None
        self.disbeliever_dict = None

        parent_directory = path.join('.', 'human-subjects-data')

        bdm_file = path.join(parent_directory, 'bdm.json')
        oscillator_file = path.join(parent_directory, 'oscillator.json')
        disbeliever_file = path.join(parent_directory, 'disbeliever.json')

        with open(bdm_file, 'r') as f:
            self.bdm_dict = json.load(f)
            self.num_bdm = len(self.bdm_dict['Alpha'])

        with open(oscillator_file, 'r') as f:
            self.oscillator_dict = json.load(f)
            self.num_oscillators = len(self.oscillator_dict['Alpha'])

        with open(disbeliever_file, 'r') as f:
            self.disbeliever_dict = json.load(f)
            self.num_disbelievers = len(self.disbeliever_dict['Alpha'])

    def generate_noise(self):
        return self.rng.normal(loc=0., scale=5.0)

    def generate(self):
        # Choose whether to select a bdm, a disbeliever, or an oscillator
        choice = self.rng.choice(3, p=[prob_bdm, prob_disbeliever, prob_oscillator])

        # Clear the params
        self.params = {}
        # If BDM
        if choice == 0:
            index = self.rng.choice(self.num_bdm)
            for key in self.bdm_dict.keys():
                self.params[key] = self.bdm_dict[key][index]
                if self.add_noise:
                    self.params[key] += self.generate_noise()
                    self.params[key] = max(self.params[key], 2.0)
        # If disbeliever
        elif choice == 1:
            index = self.rng.choice(self.num_disbelievers)
            for key in self.disbeliever_dict.keys():
                self.params[key] = self.disbeliever_dict[key][index]
                if self.add_noise:
                    self.params[key] += self.generate_noise()
                    self.params[key] = max(self.params[key], 2.0)
        # If oscillator
        elif choice == 2:
            index = self.rng.choice(self.num_oscillators)
            for key in self.oscillator_dict.keys():
                self.params[key] = self.oscillator_dict[key][index]
                if self.add_noise:
                    self.params[key] += self.generate_noise()
                    self.params[key] = max(self.params[key], 2.0)

        self.params_list = [self.params['Alpha'], self.params['Beta'], self.params['ws'], self.params['wf']]
        return self.params_list
