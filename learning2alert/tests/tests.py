# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

from data_collector import calc_sample_prob_weights
from state_extractor import calc_x_to_area, calc_mean_per_area, calc_mean_per_area_and_horizon, \
    group_maintenance, data_by_area_and_horizon_to_df
from models_keras import MSEWeighted, rmse_for_ones_per_col, rmse_for_zeros_per_col, rmse_for_ones, \
    rmse_for_zeros, BinaryCrossEntropyWeighted
import tensorflow as tf
import unittest
import numpy as np
import utils
from state_extractor import StateExtractor
import pandas as pd
import pre_process_data_utils

class TestStateExtractorUtils(unittest.TestCase):

    def test_x_to_area(self):
        x_to_subid = np.array([3, 4, 1, 0, 2])
        sub_id_by_area = [
            np.array([0, 3]), #  area 0
            np.array([4, 1, 2])] # area 1
        x_to_area = calc_x_to_area(x_to_subid, sub_id_by_area)

        expected_x_to_area = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(x_to_area, expected_x_to_area)

    def test_calc_mean_per_area(self):
        x = np.array([5., 0, -2, 4, 3])
        x_to_area = np.array([0, 1, 0, 0, 1]) # 0th element of x belongs to area 0, 1st to area 1, 2nd to area 0 etc
    
        mean_per_area = calc_mean_per_area(x, x_to_area)
        expected_mean_per_area = np.array([(5 - 2 + 4) / 3, (0 + 3) / 2 ])
        np.testing.assert_array_equal(mean_per_area, expected_mean_per_area)
        
        add_mask = np.array([1, 1, 1, 0, 1]) # filter out the 3rd element of x
        mean_per_area_mask = calc_mean_per_area(x, x_to_area, additional_mask=add_mask)
        expected_mean_per_area_mask = np.array([(5 - 2) / 2, (0 + 3) / 2 ])
        np.testing.assert_array_equal(mean_per_area_mask, expected_mean_per_area_mask)

    def test_calc_mean_per_area_and_horizon(self):
        x = np.array([[1., -3, 4, 2], [0, 6, 2, -1]])
        x_to_area = np.array([0, 1, 0, 1])
    
        mean_per_area = calc_mean_per_area_and_horizon(x, x_to_area)
        expected_mean_per_area = np.array([
            [(1 + 4) / 2, (-3 + 2) / 2 ], 
            [(0 + 2) / 2, (6 - 1) / 2]])
        np.testing.assert_array_equal(mean_per_area, expected_mean_per_area)
        
        add_mask = np.array([1, 1, 1, 0]) # filter out last element from x
        mean_per_area_mask = calc_mean_per_area_and_horizon(x, x_to_area, additional_mask=add_mask)
        expected_mean_per_area_mask = np.array([
            [(1 + 4) / 2, -3 ], 
            [(0 + 2) / 2, 6]])
        np.testing.assert_array_equal(mean_per_area_mask, expected_mean_per_area_mask)


    def test_group_maintenance(self):
        maintenance = np.array([
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
        ])
        # Assert grouping works for 2 groups
        grouped_maintenance = group_maintenance(maintenance, n_groups=2)
        np.testing.assert_array_equal(
            grouped_maintenance,
            np.array([
                [1, 0, 1, 0],
                [1, 0, 1, 1],
            ])
        )
        # Assert nothing happens when n_groups == number of rows
        grouped_maintenance = group_maintenance(maintenance, n_groups=5)
        np.testing.assert_array_equal(
            grouped_maintenance,
            maintenance
        )


    def test_data_by_area_and_horizon_to_df(self):
        """"""
        data = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        df = data_by_area_and_horizon_to_df(data, title="power", area_indices=[0, 1])
        np.testing.assert_array_equal(df.values, np.array([[1, 3, 5, 2, 4, 6]]))

        df2 = data_by_area_and_horizon_to_df(data, title="power", area_indices=[1])
        np.testing.assert_array_equal(df2.values, np.array([[2, 4, 6]]))



class TestDataCollectorUtils(unittest.TestCase):

    def test_calc_sample_prob_weights(self):
        n_each_week_day = np.array([3, 1, 0, 2])
        probs = calc_sample_prob_weights(n_each_week_day)
        expected = [1 / 5.5, 3 / 5.5, 1,  1.5 / 5.5]
        np.testing.assert_array_equal(probs, expected)

        n_each_week_day = np.array([0, 5, 0, 0])
        probs = calc_sample_prob_weights(n_each_week_day)
        expected = [1, 0, 1, 1]
        np.testing.assert_array_equal(probs, expected)
        

    


class TestModelUtils(unittest.TestCase):

    def test_mse_weighted(self):
        y_true = tf.constant([
            [0, 1., 1], 
            [1, 0, 1]])
        y_pred = tf.constant([
            [1/10, 2/5, 1/3], 
            [9/10, 1/2, 0]])
        ratio_ones = tf.constant([2/3, 1/4, 3/5])
        loss = MSEWeighted(ratio_ones).mse_loss(y_true, y_pred)
        # "manual" calculation of expected loss:
        # mask_ones = [[0, 3/4, 2/5], 
        #              [1/3, 0, 2/5]]
        # mask_zeros = [[2/3, 0, 0], 
        #               [0, 1/4, 0]]
        # mask_tot = [[2/3, 3/4, 2/5], 
        #             [1/3, 1/4, 2/5]]
        # abs_diff = [[1/10, 3/5, 2/3], 
        #             [1/10, 1/2, 1]]
        # element_wise_loss = abs_diff**2 * mask_tot:
        element_wise_loss = np.array([
            [1/100 * 2/3, 9/25 * 3/4, 4/9 * 2/5], 
            [1/100 * 1/3, 1/4 * 1/4, 1 * 2/5]])
        expected_loss = element_wise_loss.mean()
        assert abs(loss.numpy() - expected_loss) < 1e-7 , "weighted mse not producing expected loss"

    def test_rmse_per_col(self):
        y_true = tf.constant([
            [1., 0, 0], 
            [1, 0, 1]])
        y_pred = tf.constant([
            [0.9, 0.1, 0.5], 
            [0.8, 0.9, 0.5]])
        expected_rmse_ones = tf.sqrt(tf.constant([
            ((1 - 0.9)**2 + (1 - 0.8)**2) / 2,
            0,
            (1 - 0.5)**2,
            ]))
        expected_rmse_zeros = tf.sqrt(tf.constant([
            0,
            (0.1**2 + 0.9**2) / 2,
            0.5**2,
            ]))

        rmse_ones = rmse_for_ones_per_col(y_true, y_pred).numpy()
        rmse_zeros = rmse_for_zeros_per_col(y_true, y_pred).numpy()

        np.testing.assert_array_almost_equal(rmse_ones, expected_rmse_ones.numpy(), decimal=1e-6)
        np.testing.assert_array_almost_equal(rmse_zeros, expected_rmse_zeros.numpy(), decimal=1e-6)

        y_true = tf.constant([
            [0., 0, 0], 
            [0, 0, 0]])

        expected_rmse_ones = tf.constant([0., 0, 0])
        rmse_ones = rmse_for_ones_per_col(y_true, y_pred).numpy()
        np.testing.assert_array_equal(rmse_ones, expected_rmse_ones.numpy())

    def test_rmse(self):
        """"""
        y_true = tf.constant([
            [0., 1], 
            [1, 1]])
        y_pred = tf.constant([
            [0.9, 0.1], 
            [0.8, 0.9]])
        rmse_zeros = rmse_for_zeros(y_true, y_pred)
        assert rmse_zeros == 0.9

    def test_cross_entropy_weighted(self):
        y_true = tf.constant([
            [0., 1, 1], 
            [0, 1, 0]])
        y_pred = tf.constant([
            [0.1, 0.8, 0.3], 
            [0.3, 0.6, 0.4]])

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        loss_expected = bce(y_true, y_pred).numpy()

        fraction_ones = np.array([0.5, 0.5, 0.5])
        loss = BinaryCrossEntropyWeighted(fraction_ones).bce_loss(y_true, y_pred).numpy()
        assert abs(loss - loss_expected) < 1e-6, "Custom binary cross entropy function not matching tf.keras function"




class StateExtractorTest(StateExtractor):
    """
    Dummy state extractor used for test
    """
    def extract_state(self, obs, exp_id: int = 0) -> pd.DataFrame:
        return pd.DataFrame([[0, 0]], columns=["rho_0", "rho_1"])


class TestUtils(unittest.TestCase):

    """def test_filter_and_reorder(self):
        """"""
        X = pd.DataFrame(
            data=[[10., 11, 12], [1, 2, 3]],
            columns=["hour", "rho_1", "rho_0"]
        )
        X_out = utils.filter_and_reorder_X(X, StateExtractorTest, "l2rpn_idf_2023")

        X_expected_out = pd.DataFrame(
            data=[[12, 11], [3, 2]],
            columns=["rho_0", "rho_1"]
        )
        pd.testing.assert_frame_equal(X_out, X_expected_out)"""


    def test_calc_confusion_metrics(self):
        
        y_true = pd.DataFrame(data=[[1., 0], [1, 0], [0, 1]], columns=["cont_0", "cont_1"])
        y_pred = np.array([[1., 1], [1, 0], [1, 1]])

        expected_conf_per_col = np.array([[1., 0, 0, 1], [1, 0, 0.5, 0.5]])
        conf_per_col = utils.calc_confusion_metrics_per_col(y_true, y_pred)
        np.testing.assert_array_equal(conf_per_col.values, expected_conf_per_col)

        expected_conf = np.array([1., 0, 1/3, 2/3]).round(3)
        conf = utils.calc_confusion_metrics(y_true, y_pred).round(3)
        np.testing.assert_array_equal(conf.values[0], expected_conf)
    

    """def test_multiclass_f1(self):

        y_true = pd.DataFrame(data=[[1., 0], [1, 0], [0, 1]], columns=["cont_0", "cont_1"])
        y_pred = np.array([[1., 1], [1, 0], [1, 1]])
        score = utils.multiclass_f1(y_true, y_pred)

        # TODO: write assertion test"""

    def test_survival_timestep_to_survival(self):
        cols = [f"cont_{c}_{t}_0" for c in range(2) for t in range(3)]
        Y_survival_t = pd.DataFrame([
            [1, 1, 1] + [1, 1, 0],
            [0, 0, 0] + [1, 0, 0],
            [1, 0, 0] + [1, 1, 1],
            ],
            columns=cols,
        )
        data_survival = utils.survival_timestep_to_survival(Y_survival_t, horizon=2, n_cont=2)
        np.testing.assert_array_equal(
            data_survival,
            np.array([
                [1, 0],
                [0, 0],
                [0, 1],
            ])
        )



class TestDataProcessing(unittest.TestCase):

    def test_process_Y_failure_t_data(self):
        # Inputs
        data = np.array([
            [1, 3, 0],
            [-1, -1, -1]])
        horizon = 3
        df_failure_t = pd.DataFrame(data=data, columns=[f"cont_{i}" for i in range(3)])
        # Process data
        df_features = pre_process_data_utils.process_Y_failure_t_data(df_failure_t, horizon)
        # Expected output
        expected_val = np.array([
           [1, 0, 0, 0] + [1, 1, 1, 0] + [0, 0, 0, 0],
           np.ones(12),
        ])
        np.testing.assert_array_equal(df_features.values, expected_val)
    
    def test_make_Y_1D(self):
        # Define inputs
        X = pd.DataFrame(
            data=np.array([
                [2., 3], 
                [5, 4]]),
            columns=["feat_0", "feat_1"],)
        Y = pd.DataFrame(
            data=np.array([
                [1., 0, 1], 
                [0, 1, 0]]),
            columns=["survival_cont_0", "survival_cont_1", "survival_cont_2"],)
        # Transform data
        X_new, Y_new = pre_process_data_utils.make_Y_1D(X, Y)
        # Check that output is as expected
        X_expected = np.array([
            [2, 3, 1, 0, 0],
            [2, 3, 0, 1, 0],
            [2, 3, 0, 0, 1],
            [5, 4, 1, 0, 0],
            [5, 4, 0, 1, 0],
            [5, 4, 0, 0, 1],
        ])
        Y_expected = np.array([1, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(X_new.values, X_expected)
        np.testing.assert_array_equal(Y_new.values, Y_expected)


    def test_downsample_Y_containing_only_ones(self):
        """
        Test that downsampling works as expected
        """
        data_x = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
        ])
        data_y = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])
        X = pd.DataFrame(data_x, columns=["feat0", "feat1"])
        Y = pd.DataFrame(data_y, columns=["cont0", "cont1", "cont2"])
        rng = np.random.default_rng(seed=0)

        X_expected = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [9, 10],
            [11, 12],
        ])
        Y_expected = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])

        X_expected2 = np.array([
            [1, 2],
            [5, 6],
            [9, 10],
            [11, 12],
        ])
        Y_expected2 = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])

        # 3 / 6 samples contain only 1s, drop one -> new ratio is 2 / 5
        X_d, Y_d = pre_process_data_utils.downsample_Y_containing_only_ones(X, Y, 2/5, rng=rng)
        np.testing.assert_array_equal(X_d.values, X_expected)
        np.testing.assert_array_equal(Y_d.values, Y_expected)

        # drop two samples -> new ratio is 1 / 4 
        X_d2, Y_d2 = pre_process_data_utils.downsample_Y_containing_only_ones(X, Y, 1/4, rng=rng)
        np.testing.assert_array_equal(X_d2.values, X_expected2)
        np.testing.assert_array_equal(Y_d2.values, Y_expected2)


    def test_get_Y_survival_for_area(self):
        
        data = np.array([
            [1, 3, 5],
            [2, 4, 6],
        ])
        col_ids = [3, 5, 0]
        line_ids_area = [0, 3]
        line_ids_area = [5]

        Y = pd.DataFrame(
            data=data,
            columns=[f"cont_{i}" for i in col_ids],
        )
        np.testing.assert_array_equal(
            pre_process_data_utils.get_Y_survival_for_area(Y, [0, 3]).values,
            np.array([
                [1, 5],
                [2, 6]
            ])
        )
        np.testing.assert_array_equal(
            pre_process_data_utils.get_Y_survival_for_area(Y, [5]).values,
            np.array([
                [3],
                [4]
            ])
        )
        
        

        



idx = [106, 93, 88, 162, 68, 117, 180, 160, 136, 141, 131, 121, 125, 126, 110, 154, 81, 43, 33, 37, 62, 61]

cols = [""]



# Test that pickling obs works as intended
# TODO: make into test
"""
env = grid2op.make("l2rpn_idf_2023")
env.set_id(42)
obs = env.reset()
load_p_hat, load_q_hat, gen_p_hat, gen_v_hat, maintenance = obs.get_forecast_arrays()

dict_before = {
    "rho": obs.rho,
    "load_p_hat": load_p_hat,
    "load_q_hat": load_q_hat,
    "gen_p_hat": gen_p_hat,
    "gen_v_hat": gen_v_hat,
    "maintenance": maintenance,
}
pickle.dump(obs, open("obs.p", "wb"))
loaded_obs = pickle.load(open("obs.p", "rb"))
load_p_hat2, load_q_hat2, gen_p_hat2, gen_v_hat2, maintenance2 = loaded_obs.get_forecast_arrays()

dict_after = {
    "rho": loaded_obs.rho,
    "load_p_hat": load_p_hat2,
    "load_q_hat": load_q_hat2,
    "gen_p_hat": gen_p_hat2,
    "gen_v_hat": gen_v_hat2,
    "maintenance": maintenance2,
}
for key in dict_before.keys():
    np.testing.assert_array_equal(
        dict_before[key],
        dict_after[key]
    )
"""

if __name__ == '__main__':
    unittest.main()
    #TestDataProcessing().test_get_Y_survival_for_area()
