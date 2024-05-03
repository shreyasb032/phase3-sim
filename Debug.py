import numpy as np

num_sites = 3
current_idx = 0
number_of_sites_to_go = num_sites - current_idx
value_matrix = np.zeros((
    number_of_sites_to_go + 1,  # stages
    number_of_sites_to_go + 1,  # possible health
    number_of_sites_to_go + 1  # possible time
))

action_matrix = np.zeros((
    number_of_sites_to_go,
    number_of_sites_to_go,
    number_of_sites_to_go
))

for stage in reversed(range(number_of_sites_to_go)):
    possible_health = 100 - np.arange(stage + 1) * 10
    possible_time = 100 - np.arange(stage + 1) * 10
    print(stage)
    print(possible_health)
    print(possible_time)
    print()