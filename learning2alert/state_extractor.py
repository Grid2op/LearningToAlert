# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

import numpy as np
import pandas as pd
from grid2op.dtypes import dt_bool
from typing import Tuple

class StateExtractor:
    """
    Class used to get a state from an observation.
    The state is a dataframe formatted for model learning via e.g keras and scikit learn.
    """
    def __init__(self, env) -> None:
        pass

    def extract_state(self, obs, exp_id: int=0) -> pd.DataFrame:
        """Extract state from observation"""
        raise NotImplementedError



class Rho(StateExtractor):
    """
    Class used for extracting rho from an observation.
    """    
    def extract_state(self, obs, exp_id=0):
        return pd.DataFrame([obs.rho], columns=[f"rho_{i}" for i in range(len(obs.rho))], index=[exp_id])


class Dates(StateExtractor):
    """
    Extract dates from obs. Mostly useful for data analysis.
    """
    def extract_state(self, obs, exp_id: int = 0) -> pd.DataFrame:
        return pd.DataFrame({
            "month": obs.month,
            "day_of_month": obs.day,
            "day_of_week": obs.day_of_week,
            "hour_of_day": obs.hour_of_day,
            "minute_of_hour": obs.minute_of_hour,
        }, index=[exp_id]) 


class RhoMaintenanceDates(StateExtractor):
    """
    Class used for extracting features based on rho, forecasted maintenance, datetime features and more.
    """
    def __init__(self, env, area=None, horizon=12, maintenance_line_name=None) -> None:
        self.lines_by_area = list(env._game_rules.legal_action.lines_id_by_area.values())
        self.n_areas = len(self.lines_by_area)
        self.horizon = horizon
        if area is not None:
            self.areas = [area]
        else:
            self.areas = [i for i in range(self.n_areas)]

        # if env name not one of these 3, raise error since maintenance lines are hard-coded.
        if env.name != "l2rpn_idf_2023" and env.name != "input_data_local" and env.name != "input_data_test" and maintenance_line_name is None:
            raise ValueError(
                "For environments other than 'l2rpn_idf_2023' and 'input_data_local', maintenance_line_name must be \
                specified.")
        
        if maintenance_line_name is None:
            self._maintenance_line_names = [
                "26_31_106", "21_22_93", "17_18_88", "4_10_162", "12_14_68",
                "29_37_117","62_58_180", "62_63_160", "48_50_136", "48_53_141",
                "41_48_131", "39_41_121", "43_44_125", "44_45_126", "34_35_110",
                "54_58_154", "74_117_81", "80_79_175", "93_95_43", "88_91_33",
                "91_92_37"]
        else:
            self._maintenance_line_names = maintenance_line_name
        
        self.maintenance_line_idx = np.where(np.in1d(env.name_line, self._maintenance_line_names))[0]
        # Test that maintenance_line_idx was extracted correctly:
        assert_arrays_contain_same_data(self._maintenance_line_names, env.name_line[self.maintenance_line_idx])
        if len(self.areas) == 1:
            # Filter maintenance_line_idx to only include for one area
            self.maintenance_line_idx = np.intersect1d(self.lines_by_area[self.areas[0]], self.maintenance_line_idx)


    def _extract_maintenance_features(self, obs, exp_id) -> pd.DataFrame:
        n_groups = 3 # summarize timesteps into this many categories
        _, _, _, _, maintenance = obs.get_forecast_arrays()
        # Only include lines which can go into maintenance:
        filtered_maintenance = maintenance[:, self.maintenance_line_idx]
        if len(self.areas) != 1:
            assert maintenance.sum() == filtered_maintenance.sum(), "Data lost when filtering by self.maintenance_line_idx!"
        # Group to reduce dimension, then flatten
        grouped_maintenance = group_maintenance(filtered_maintenance, n_groups=n_groups)
        data = grouped_maintenance.flatten("F")
        columns = [f"maintenance_{line}_group_{g}" for line in self.maintenance_line_idx for g in range(n_groups)]
        return pd.DataFrame(
            data=[data],
            columns=columns,
            index=[exp_id]
        )


    def _extract_rho_features(self, obs, exp_id) -> pd.DataFrame:
        """
        Create dataframe from obs.rho. If self.area specified, filter.
        """
        if len(self.areas) == 1:
            rho = obs.rho[self.lines_by_area[self.areas[0]]]
            return pd.DataFrame(
                [rho], 
                columns=[f"rho_{i}" for i in self.lines_by_area[self.areas[0]]],
                index=[exp_id])
        else:
            return pd.DataFrame([obs.rho], columns=[f"rho_{i}" for i in range(len(obs.rho))], index=[exp_id])


    def extract_state(self, obs, exp_id=0):
        """
        Extract state from observation
        """
        df_maintenance = self._extract_maintenance_features(obs, exp_id)
        # nof lines going into maintenance within horizon steps + 1
        idx_lines_maintenance_in_horizon = np.where(np.logical_and(obs.time_next_maintenance > 0, obs.time_next_maintenance <= self.horizon + 1))
        n_lines_maintenance_in_horizon = len(idx_lines_maintenance_in_horizon[0])
        
        # Get rho features
        df_rho = self._extract_rho_features(obs, exp_id)

        df_scalars = pd.DataFrame({
            "n_lines_maintenance_in_horizon": n_lines_maintenance_in_horizon, # lines with upcoming maintenance soon
            "n_disconnect_lines": (obs.rho == 0).sum(),
            # diff in nof nodes in energy graph now vs in original state:
            "n_extra_nodes": len(obs.get_energy_graph().nodes) - obs.n_sub, # NOTE: this computation is very slow - bottleneck!
            "sum_timestep_overflow": obs.timestep_overflow.sum(),
            "sum_time_before_cd_line": obs.time_before_cooldown_line.sum(),
        }, index=[exp_id])
       
        df = pd.concat([
            df_rho, 
            df_maintenance,
            process_time_features(obs.month, 12, "month", exp_id), # 1 to 12
            process_time_features(obs.day, 30.44, "day_of_month", exp_id), # 1 to 28, 30 or 31
            process_time_features(obs.day_of_week, 7, "day_of_week", exp_id), # 0 to 6
            process_time_features(obs.hour_of_day, 24, "hour_of_day", exp_id), # 
            process_time_features(obs.minute_of_hour, 60, "minute_of_hour", exp_id),
            df_scalars,
            ], axis=1)

        assert df.shape[0] == 1, f"'df.shape[0]'=={df.shape[0]}, expected 1"
        return df



class RhoMaintenanceDatesPower(RhoMaintenanceDates):
    """
    Class used for extracting features based on rho, forecasted maintenance, datetime features, real and forecasted 
    load, generation and renewable generation per area.
    """
    def __init__(self, env, area=None, horizon=12, maintenance_line_name=None) -> None:
        super().__init__(env, area, horizon, maintenance_line_name)
        self.sub_id_by_area = list(env._game_rules.legal_action.substations_id_by_area.values()) 
        self.gen_to_area = calc_x_to_area(env.gen_to_subid, self.sub_id_by_area)
        self.load_to_area = calc_x_to_area(env.load_to_subid, self.sub_id_by_area)
        self.gen_renewable = env.gen_renewable

    def _extract_p(self, obs, exp_id):
        # Create load and gen states
        load_p_hat, load_q_hat, gen_p_hat, gen_v_hat, maintenance = obs.get_forecast_arrays()

        idx_gen_non_renew = self.gen_renewable == False 
        idx_gen_renew = self.gen_renewable == True

        load_p_mean = calc_mean_per_area(obs.load_p, self.load_to_area)
        gen_p_mean = calc_mean_per_area(obs.gen_p, self.gen_to_area, idx_gen_non_renew)
        gen_p_renew_mean = calc_mean_per_area(obs.gen_p, self.gen_to_area, idx_gen_renew)

        load_p_hat_mean = calc_mean_per_area_and_horizon(load_p_hat, self.load_to_area)
        gen_p_hat_mean = calc_mean_per_area_and_horizon(gen_p_hat, self.gen_to_area, idx_gen_non_renew)
        gen_p_hat_renew_mean = calc_mean_per_area_and_horizon(gen_p_hat, self.gen_to_area, idx_gen_renew)

        load_p_conc = np.concatenate([[load_p_mean], load_p_hat_mean])
        gen_p_conc = np.concatenate([[gen_p_mean], gen_p_hat_mean])
        gen_p_renew_conc = np.concatenate([[gen_p_renew_mean], gen_p_hat_renew_mean])

        df_load = data_by_area_and_horizon_to_df(load_p_conc, "load_p", self.areas, exp_id)
        df_gen =  data_by_area_and_horizon_to_df(gen_p_conc, "gen_p", self.areas, exp_id)
        df_gen_renew = data_by_area_and_horizon_to_df(gen_p_renew_conc, "gen_p_renew", self.areas, exp_id)
        return df_load, df_gen, df_gen_renew

    def extract_state(self, obs, exp_id=0):
        """
        Extract state from observation
        """
        df = super().extract_state(obs, exp_id)
        # Get power features
        df_load, df_gen, df_gen_renew = self._extract_p(obs, exp_id)
        # Taking some extra steps to ensure df_load, df_gen, df_gen_renew is placed in the same
        # order as before updating code. This is simply to be able to load old trained models, and
        # can be replaced with this line later:
        # df = pd.concat([df, df_load, df_gen, df_gen_renew], axis=1)
        rho_columns = df.filter(regex="rho").columns
        df_rho = df[rho_columns]
        df = df.drop(rho_columns, axis=1)
        df = pd.concat([df_rho, df_load, df_gen, df_gen_renew, df], axis=1)
        assert df.shape[0] == 1, f"'df.shape[0]'=={df.shape[0]}, expected 1"
        return df


# -------------------- UTILS --------------------------- #


def calc_x_to_area(x_to_id, id_by_area):
    x_to_area = np.zeros_like(x_to_id)
    for i, id in enumerate(x_to_id):
        for area, id_list in enumerate(id_by_area):
            if id in id_list:
                x_to_area[i] = area
                break
    return x_to_area

def calc_mean_per_area(array, idx_to_area, additional_mask=None):
    mean_per_area = []
    n_areas = idx_to_area.max() + 1
    for i in range(n_areas):
        mask = idx_to_area == i
        if additional_mask is not None:
            mask *= np.array(additional_mask, dtype=bool)
        mean_per_area.append(np.mean(array[mask]))
    return np.array(mean_per_area)

def calc_mean_per_area_and_horizon(array, idx_to_area, additional_mask=None):
    mean_per_area = []
    n_areas = idx_to_area.max() + 1
    for i in range(n_areas):
        mask = idx_to_area == i
        if additional_mask is not None:
            mask *= np.array(additional_mask, dtype=bool)
        mean_per_area.append(np.mean(array[:, mask], axis=1))
    return np.array(mean_per_area).T

def data_by_area_and_horizon_to_df(array: np.ndarray, title: str, area_indices: list, exp_id: int):
    n_rows = array.shape[0]
    df_list = []
    for area in area_indices:
        data = array[:, area]
        columns = [f"{title}_step_{i}_area_{area}" for i in range(n_rows)]
        df_list.append(pd.DataFrame([data], columns=columns, index=[exp_id]))
    return pd.concat(df_list, axis=1)


def process_time_features(time_feat: np.array, period: float, label: str, exp_id: int) -> pd.DataFrame:
    """
    Turn time features into sin and cos features
    """
    sin_feat = np.sin(2 * np.pi * time_feat / period)
    cos_feat = np.cos(2 * np.pi * time_feat / period)
    return pd.DataFrame([[sin_feat, cos_feat]], columns=[f"{label}_sin", f"{label}_cos"], index=[exp_id])


def assert_arrays_contain_same_data(arr1, arr2):
    """Assert that arrays contain same elements, possibly in different orders."""
    arr1_sorted = np.sort(arr1)
    arr2_sorted = np.sort(arr2)
    np.testing.assert_array_equal(arr1_sorted, arr2_sorted)



def group_maintenance(maintenance, n_groups):
    """
    Group upcoming maintenance data into chunks of time.

    maintenace should be a (time_horizon + 1) x n_lines array, with a 1 to indicate maintenance, and 0 to indicate no maintenance.
    maintenance is reduced in size along time-axis (0). First its split into n_groups. If there is maintenance at any timestep
    in the group for that line, the value is set to 1, else 0. Output is an array of shape n_groups x n_lines.

    Example:
    >> maintenance = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
    ])
    >> group_maintenance(maintenance, n_groups=2)
    array([[1, 0, 0],
           [1, 0, 1]])
    """
    assert n_groups <= maintenance.shape[0], f"n_groups ({n_groups}) must be <= the dimension of maintenance along axis=0 ({maintenance.shape[0]})"
    # Split along time axis:
    split_maintenance = np.array_split(maintenance, n_groups, axis=0)
    features = []
    for m in split_maintenance:
        # If maintenance at any timestep for line in group, set value to 1
        features.append((m == 1).any(axis=0))
    features = np.array(features) * 1. # convert to float array
    return features

# --------------------------------------------------------------- #
