# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

import os
import numpy as np
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict
import pickle
import sklearn.metrics
import matplotlib.pyplot as plt
import sys
import copy

def make_dir_if_doesnt_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_params(dict, savedir):
    dict = copy.deepcopy(dict) # copy dict so we dont modify it globally
    # Turn values into str, unless dictionary, where inner vals are turned into str
    for key, val in dict.items():
        if isinstance(val, Dict):
            for key2, val2 in val.items():
                if type(val2) is not str:
                    dict[key][key2] = str(val2)
        elif type(val) is not str:
            dict[key] = str(val)
    with open(savedir, 'w') as outfile:
        yaml.dump(dict, outfile, default_flow_style=False)


def calc_fraction_ones(y: np.ndarray) -> float:
    """
    Calculate fraction of 1:s in y. Assumes y only contains 0:s and 1:s.
    """
    return y.sum() / y.size

def calc_fraction_ones_per_col(y: np.ndarray) -> np.ndarray:
    """
    Calculate fraction of 1:s in y. Assumes y only contains 0:s and 1:s.
    """
    return y.sum(axis=0) / y.shape[0]


def calc_true_positive_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the ratio TP / P, where TP is true positives, and P the number of positive samples in y_true.
    """
    assert type(y_true) == np.ndarray, "y_true must have type np.ndarray"
    assert type(y_pred) == np.ndarray, "y_pred must have type np.ndarray"
    ones_at = y_true == 1
    true_pos_acc = ((y_pred == 1) * ones_at).sum() / ones_at.sum()
    return true_pos_acc


def calc_true_negative_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the ratio TN / N, where TN is true negatives, and N the number of negative samples in y_true.
    """
    assert type(y_true) == np.ndarray, "y_true must have type np.ndarray"
    assert type(y_pred) == np.ndarray, "y_pred must have type np.ndarray"
    zeros_at = y_true == 0
    true_neg_acc = ((y_pred == 0) * zeros_at).sum() / zeros_at.sum()
    return true_neg_acc


def calc_true_positive_ratio_per_col(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the ratio TP / P per column, where TP is true positives, and P the number of positive samples in y_true.
    """
    assert type(y_true) == np.ndarray, "y_true must have type np.ndarray"
    assert type(y_pred) == np.ndarray, "y_pred must have type np.ndarray"
    ones_at = y_true == 1
    true_pos_acc_per_col = ((y_pred == 1) * ones_at).sum(axis=0) / ones_at.sum(axis=0)
    return true_pos_acc_per_col


def calc_true_negative_ratio_per_col(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the ratio TN / N per column, where TN is true negatives, and N the number of negative samples in y_true.
    """
    assert type(y_true) == np.ndarray, "y_true must have type np.ndarray"
    assert type(y_pred) == np.ndarray, "y_pred must have type np.ndarray"
    zeros_at = y_true == 0
    true_neg_acc_per_col = ((y_pred == 0) * zeros_at).sum(axis=0) / zeros_at.sum(axis=0)
    return true_neg_acc_per_col


def calc_confusion_metrics_per_col(y_true: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculate ratios of confusion metrics per column.
    """
    true_pos_acc_per_col = calc_true_positive_ratio_per_col(y_true.values, y_pred)
    false_neg_acc_per_col = 1 - true_pos_acc_per_col

    true_neg_acc_per_col = calc_true_negative_ratio_per_col(y_true.values, y_pred)
    false_pos_acc_per_col = 1 - true_neg_acc_per_col

    data = np.array([true_pos_acc_per_col, false_neg_acc_per_col, true_neg_acc_per_col, false_pos_acc_per_col]).T
    return pd.DataFrame(
        data,
        columns=["true_pos", "false_neg", "true_neg", "false_pos"],
        index=y_true.columns,
    )

def calc_confusion_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Calculate ratios of confusion metrics.
    """
    true_pos_acc = calc_true_positive_ratio(y_true.values, y_pred)
    false_neg_acc = 1 - true_pos_acc

    true_neg_acc = calc_true_negative_ratio(y_true.values, y_pred)
    false_pos_acc = 1 - true_neg_acc

    data = [true_pos_acc, false_neg_acc, true_neg_acc, false_pos_acc]
    return pd.DataFrame(
        [data],
        columns=["true_pos", "false_neg_acc", "true_neg", "false_pos"],
    )


def calc_sample_weights(y_true, fraction_ones, zeros_mult_factor: float=1.):
    """
    Calc sample weights.
    Should work for pd.DataFrame, np.array, tf.Tensor
    """
    sample_weights_ones = y_true * (1 - fraction_ones)
    sample_weights_zeros = (1 - y_true) * fraction_ones * zeros_mult_factor
    sample_weights = sample_weights_ones + sample_weights_zeros
    return sample_weights


def load_data(
    load_dir: str, 
    val: bool=True,
    seed: int=0,
    ) -> Dict[str, pd.DataFrame]:
    """
    Read train and test data from load_dir. Optionally, split train into train and validation dataset.
    """

    # Read data
    X_train = pd.read_csv(f"{load_dir}/X_train.csv", index_col=0)
    Y_train = pd.read_csv(f"{load_dir}/Y_train.csv", index_col=0)
    X_test = pd.read_csv(f"{load_dir}/X_test.csv", index_col=0)
    Y_test = pd.read_csv(f"{load_dir}/Y_test.csv", index_col=0)

    X_val = None
    Y_val = None
    if val:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, shuffle=True, random_state=seed)
        np.testing.assert_array_equal(X_val.index.values, Y_val.index.values)
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Y_test": Y_test,
    }


def load_pickled_model(path: str):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model

def pickle_model(model, path: str):
    with open(path, "wb") as file:
        pickle.dump(model, file)



def balanced_accuracy_multiclass(y_true: pd.DataFrame, y_pred: np.ndarray, thresholds=0.5, weights=None) -> float:
    """
    Calculated the (weighted) balanced accuracy: w0 * TP + w1* TN, where TP is true positives, TN true negatives,
    and w0, w1 are weights such that w0 + w1 = 1. Default values are w0 = w1 = 1/2. 
    """
    try:
        y_true = y_true.values # for pd.DataFrame, get values
    except AttributeError:
        pass
    if weights is None:
        weights = np.array([1/2, 1/2])
    assert len(weights) == 2, "Len of weights should be 2"
    assert np.sum(weights).round(3) == 1, "weights must sum to 1 (within a precision of 3 decimals)"
    y_pred_class = (y_pred > thresholds) * 1.
    tp = calc_true_positive_ratio_per_col(y_true, y_pred_class).mean()
    tn = calc_true_negative_ratio_per_col(y_true, y_pred_class).mean()
    return weights[0] * tp + weights[1] * tn


def survival_timestep_to_survival(Y_survival_t: pd.DataFrame, horizon=12, n_cont=22) -> np.array:
    """
    Convert survival at timestep per contingency, into pred of survival per contingency.
    Only if the grid survives (value 1) for all timesteps for a contingency, survival is set to 1 for that contingency in that sample, else 0.
    
    Example:
    >> Y_survival_t = pd.DataFrame([
        [1, 1, 1, 0],
        [1, 0, 0, 0]],
        columns=["cont_0_t_0", "cont_0_t_1", "cont_1_t_0", "cont_1_t_1"]
        )
    >> Y_survival_t
       cont_0_t_0  cont_0_t_1  cont_1_t_0  cont_1_t_1
    0           1           1           1           0
    1           1           0           0           0
    >> survival_timestep_to_survival(Y_survival_t, horizon=2, n_cont=2)
       cont_0  cont_1
    0    True   False
    1   False   False
    """
    survival_data = []
    for i, failure_t_per_cont in Y_survival_t.iterrows():
        reshaped_sample = failure_t_per_cont.values.reshape((n_cont, horizon + 1))
        survival_per_cont = (reshaped_sample == 1).all(axis=1) * 1.
        survival_data.append(survival_per_cont)
    return np.array(survival_data)


def select_thresholds(Y_val: pd.DataFrame, Y_pred_val: np.array, tn_weight: float=1., plot: bool=False):
    """
    Select thresholds for classification using ROC curves. 

    Input:
    Y_val: validation data labels, shape n_samples x n_classes, containing only 0s and 1s.
    Y_pred_val: predicted validation data probabilities, same shape as Y_val.
    tn_weight: how much to weigh true negatives over true positives when selecting threshold
    plot: save plots of roc curves for each contingency.

    Output:
    thresholds: one threshold per column (class)
    """
    selected_thresholds = np.zeros(Y_pred_val.shape[-1])
    for i in range(len(selected_thresholds)):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_val.values[:, i], Y_pred_val[:, i])
        idx_threshold = np.argmax(tpr - fpr * tn_weight) # equivalent to maximizing (true pos rate) + (true neg rate) * k
        selected_thresholds[i] = thresholds[idx_threshold]
        if plot:
            make_dir_if_doesnt_exist("figs")
            plt.figure()
            plt.plot(thresholds, 1-fpr, label="tnr")
            plt.plot(thresholds, tpr, label="tpr")
            plt.legend()
            plt.savefig(f"figs/roc_{i}.png")
            plt.close()
    return np.clip(selected_thresholds, a_min=0.05, a_max=0.95)


def load_agent_from_submission_file(env, agent_path: str):
    """
    Instantiate an agent from submission file (file must be unzipped)
    """
    # Load Javaness agent
    submission_location = os.path.join(agent_path, "submission")
    sys.path.append(agent_path)  # add agent's code and modules to python path
    from submission import make_agent  # submission module is in agent_path#
    return make_agent(env, submission_location)
