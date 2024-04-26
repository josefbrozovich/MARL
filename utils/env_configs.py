import numpy as np
import math
import itertools
import torch

def env_configs(poss_rooms=8, chose_rooms=4, goal_rooms=1):

    # we have 8 poisitions for rooms
    # 4 of those rooms are in map
    # one of those 4 rooms is the goal room

    combinations = list(itertools.combinations(range(poss_rooms), chose_rooms))

    # Initialize an empty array to store the combinations
    combinations_array = np.zeros((len(combinations), poss_rooms), dtype=int)

    # Fill the array with combinations
    for i, comb in enumerate(combinations):
        combinations_array[i, comb] = 1

    first = combinations_array.copy()
    second = combinations_array.copy()
    third = combinations_array.copy()
    fourth = combinations_array.copy()

    i = 0
    for row in first:
        index_1, index_2, index_3, index_4 = np.where(row == 1)[0]
        first[i,index_1] = 2
        second[i,index_2] = 2
        third[i,index_3] = 2
        fourth[i,index_4] = 2
        i += 1

    configs = np.concatenate((first, second, third, fourth), axis=0)

    num_rows = configs.shape[0]
    row_indices = np.random.permutation(num_rows)
    configs = configs[row_indices]

    train_index = int(0.8*np.shape(configs)[0])
    validation_index = int(0.8*np.shape(configs)[0])

    train_data = configs[:train_index,:]
    validation_data = configs[train_index:validation_index,:]
    test_data = configs[validation_index:,:]

    return train_data, validation_data, test_data
