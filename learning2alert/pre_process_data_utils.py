# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

from typing import List, Tuple
import pandas as pd
import numpy as np
import state_extractor
import grid2op
import pickle
import math
from sklearn.model_selection import train_test_split
import os
import re
from tqdm import tqdm

def combine_data(
    loadpaths: List[str], 
    new_savepath: str,
    index=None,
    ):
    """
    Load list of files, concat along axis=0 and save.
    """
    data_list = []
    for path in loadpaths:
        data_list.append(pd.read_csv(path, index_col=0))

    data_comb = pd.concat(data_list, axis=0)
    if index:
        data_comb.index = index
    else:
        data_comb.index = [i for i in range(len(data_comb))]
    data_comb.to_csv(new_savepath, sep=",")



def combine_data_XY(
    loadpaths_x: List[str], 
    loadpaths_y: List[str], 
    new_savepath_x: str, 
    new_savepath_y: str,
    ):
    """
    Load list of X and Y files, concat each along axis=0 and save.
    """
    X_list = []
    Y_list = []
    for path_x, path_y in zip(loadpaths_x, loadpaths_y):
        X_list.append(pd.read_csv(path_x, index_col=0))
        Y_list.append(pd.read_csv(path_y, index_col=0))

    X_comb = pd.concat(X_list, axis=0)
    Y_comb = pd.concat(Y_list, axis=0)

    X_comb.index = [i for i in range(len(X_comb))]
    Y_comb.index = X_comb.index

    X_comb.to_csv(new_savepath_x, sep=",")
    Y_comb.to_csv(new_savepath_y, sep=",")



def make_Y_1D(X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Turn the multi-label classification problem into a 1D problem.
    Y should have shape n_samples x n_contingencies, and each value indicate 1 for grid survival, and 0 for grid failure.

    Y is flattened, increasing the number of samples by a factor n_contingecies.
    X is copied n_contingencies times, and the contingency for which to predict success rate is added as a feature.
    """
    n_samples, n_cont = Y.shape
    # Flatten Y
    y_flat = Y.values.flatten()
    Y_new = pd.Series(data=y_flat, index=[i for i in range(len(y_flat))])
    # Stack n_samples identity matrices with shape n_cont x n_cont on top of eachother.
    # Each row is a one-hot encoding showing which contingency the corresponding Y_new belongs to.
    contingency_1hot = np.concatenate([np.identity(n_cont) for i in range(n_samples)], axis=0)
    contingency_1hot_df = pd.DataFrame(
        data=contingency_1hot,
        columns=Y.columns,
        index=Y_new.index,
    )
    # Duplicate each row in X n_cont times. Note, not the same as stacking n_cont copies of X.
    X_duplicated = X.loc[X.index.repeat(n_cont)]
    X_duplicated.index = Y_new.index
    # Add information about which contingency each sample in Y_new belongs to, to X.
    X_new = pd.concat([X_duplicated, contingency_1hot_df], axis=1)
    return X_new, Y_new


def process_Y_failure_t_data(failure_t_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Process Y-labels containing data of at which timestep the grid failed (-1 means no failure).
    Y should have shape (n_samples x n_contingencies).
    Turn each sample into a vector with dim = n_contingencies * (horizon + 1).

    For each contingency, create a vector of length (horizon + 1), and set the value to 0
    at the failure timestep. Values to the left are 1:s, values to the right are 0:s. 
    1 indicates no grid failure at timestep, and 0 indicates failure.
    The vectors for each contingency are then concatenated to create one vector per sample.
    """
    failure_data = failure_t_df.values
    n_samples, n_cont = failure_data.shape
    data = []
    for sample in range(n_samples):
        features = np.array([])
        for cont in range(n_cont):
            # Get timestep of grid collapse
            fail_t = int(failure_data[sample, cont])
            success_at_t_per_cont = np.ones(horizon + 1) # 1 indicates success in new feature
            if fail_t != -1: # -1 means no failure
                success_at_t_per_cont[fail_t] = 0 # failure at timestep fail_t
                if (fail_t + 1) < len(success_at_t_per_cont):
                    success_at_t_per_cont[fail_t+1:] = 0 # failure at all subsequent timesteps
            features = np.concatenate([features, success_at_t_per_cont])
        data.append(features)

    columns = [f"survival_t_{t}_cont_{cont}" for cont in range(n_cont) for t in range(horizon + 1)]
    return pd.DataFrame(
        data=data,
        columns=columns,
        index=failure_t_df.index)


def get_features_from_obs_list(
    obs_loadpath: str, 
    state_extractor_class: state_extractor.StateExtractor=state_extractor.RhoMaintenanceDatesPower,
    env_name: str="l2rpn_idf_2023",
    state_extractor_kwargs=None,
    ) -> pd.DataFrame:
    """
    Load observations from pickled file, use state extractor to generate dataframe.
    """
    if state_extractor_kwargs is None:
        state_extractor_kwargs = {}
    # Create env, load obs, init state extractor
    env = grid2op.make(env_name)
    with open(obs_loadpath, "rb") as file:
        obs_list = pickle.load(file)
    state_extractor = state_extractor_class(env, **state_extractor_kwargs)
    # Use state extractor to create df
    state_list = []
    for i, obs in enumerate(tqdm(obs_list)):
        # Extract state and add to list
        state = state_extractor.extract_state(obs, i)
        state_list.append(state)
    # Save state as csv
    df_state = pd.concat(state_list, axis=0)
    return df_state
    



def remove_outliers_gen_p(X, Y, max_val=500):
    """
    Remove rows in X and Y where non-renewable generated power is larger than max_val.
    """
    cols = []
    for col in X.columns:
        if "gen_p" in col and not "gen_p_renew" in col:
            cols.append(col)
    if len(cols) == 0:
        return X, Y
    X_gen = X[cols]
    index_outliers = X_gen[(X_gen > max_val).any(axis=1)].index
    return X.drop(index=index_outliers), Y.drop(index=index_outliers)


def resample_zeros(X: pd.DataFrame, Y: pd.DataFrame, desired_fraction_zeros: float=0.05, seed: int=0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeat samples in Y with the goal of increasing the number of 0s. Also attempt to balance
    out the number of 0:s between contingencies by repeating some samples more times.

    Input:
    X: feature data
    Y: label data
    desired_fraction_zeros: indirectly controls how many times samples are repeated. If each sample would contain only 
        one contingency with value 0, the resampled data would contain this fraction of 0s for each contingency. But 
        since multiple 0:s can exist, when resampling 0:s for one contingency, 0:s for another could be added. The 
        resulting fraction is therefore going to be larger.
    seed: seed for np random generator

    Output:
    X_new: resampled X
    Y_new: resamples Y
    """
    X_copies_list = []
    Y_copies_list = []
    for col in Y.columns:
        # For this contingency, get index where its 0, and nof samples
        Y_col = Y[col]
        n_current_samples = (Y_col == 0).sum()
        idx = Y_col[(Y_col == 0)].index
        # (n_current_samples + n_samples_to_add) / (n_total_samples + n_samples_to_add) = desired_fraction, solve eq:
        n_samples_to_add = (len(Y) * desired_fraction_zeros - n_current_samples) / (1 - desired_fraction_zeros)
        # Repeat indices
        nof_copies_samples = math.ceil(n_samples_to_add / n_current_samples)
        repeated_indices = idx.repeat(nof_copies_samples)
        # Save repeated data to list
        X_copies_list.append(X.loc[repeated_indices])
        Y_copies_list.append(Y.loc[repeated_indices])
    X_new = pd.concat(([X] + X_copies_list), axis=0)
    Y_new = pd.concat(([Y] + Y_copies_list), axis=0)

    X_new.index = [i for i in range(len(X_new))]
    Y_new.index = X_new.index

    # Shuffle
    gen = np.random.default_rng(seed=seed)
    idx = gen.permutation(X_new.index)
    X_new = X_new.reindex(idx)
    Y_new = Y_new.reindex(idx)
    np.testing.assert_array_equal(X_new.index.values, Y_new.index.values)
    return X_new, Y_new


def downsample_Y_containing_only_ones(
    X: pd.DataFrame, 
    Y: pd.DataFrame, 
    desired_fraction_only_ones: int=0.5,
    rng: np.random.Generator=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove some samples where all columns in Y == 1, until they make up the specified fraction.
    """
    samples_only_ones = ((Y == 1).sum(axis=1) == Y.shape[1])
    index_only_ones = samples_only_ones[samples_only_ones].index
    current_fraction_only_ones = samples_only_ones.sum() / len(Y)
    assert desired_fraction_only_ones <= current_fraction_only_ones, f"desired_fraction_only_ones \
        ({desired_fraction_only_ones}) must be smaller than or equal the current fraction of ones ({current_fraction_only_ones})"

    n_only_ones = samples_only_ones.sum()
    # (n_only_ones - n_ones_to_remove) / (n_samples - n_ones_to_remove) = desired_fraction_ones. 
    # Solve for n_ones_to_remove, and you get:
    n_ones_to_remove = (n_only_ones - desired_fraction_only_ones * len(Y)) / (1 - desired_fraction_only_ones)
    n_ones_to_remove = round(n_ones_to_remove)

    if rng:
        index_to_remove = rng.choice(index_only_ones, n_ones_to_remove, replace=False)
    else:
        index_to_remove = np.random.choice(index_only_ones, n_ones_to_remove, replace=False)

    X = X.drop(index_to_remove, axis=0)
    Y = Y.drop(index_to_remove, axis=0)
    np.testing.assert_array_equal(X.index.values, Y.index.values)
    return X, Y


def make_train_test_files(
    x_path: str, 
    y_path: str,
    save_dirname: str,
    seed: int=0,
    ):
    """
    Split data into train and test, save as 4 separate files in specified folder
    """
    # Read data
    X = pd.read_csv(x_path, index_col=0)
    Y = pd.read_csv(y_path, index_col=0)

    assert len(X) == len(Y), "X and Y does not have the same number of samples!"

    X, Y = remove_outliers_gen_p(X, Y)
    
    # Split data into train, test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=seed)

    # Assert indices match after sort
    np.testing.assert_array_equal(X_train.index.values, Y_train.index.values)
    np.testing.assert_array_equal(X_test.index.values, Y_test.index.values)

    os.mkdir(save_dirname)

    X_train.to_csv(f"{save_dirname}/X_train.csv", sep=",")
    Y_train.to_csv(f"{save_dirname}/Y_train.csv", sep=",")

    X_test.to_csv(f"{save_dirname}/X_test.csv", sep=",")
    Y_test.to_csv(f"{save_dirname}/Y_test.csv", sep=",")

    
def get_Y_survival_for_area(Y_survival: pd.DataFrame, line_ids_area: np.ndarray) -> pd.DataFrame:
    """
    Filter Y values to only include contingencies in a specific area.

    Input
    Y_survival: dataframe of shape n_samples x n_contingencies
    lines_ids_area: array of ids of lines in a given area. Used to filter out contingency ids.

    Output:
    Filtered Y_survival
    """
    pattern = "cont_([0-9]+)"
    cont_ids = []
    for col in Y_survival.columns:
        m = re.search(pattern, col)
        cont_ids.append(int(m.group(1)))

    saved_idx = []
    for i, cont_id in enumerate(cont_ids):
        if cont_id in line_ids_area:
            saved_idx.append(i)

    cols_to_keep = Y_survival.columns[saved_idx]
    return Y_survival[cols_to_keep]
