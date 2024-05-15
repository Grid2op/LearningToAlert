# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

import grid2op
from grid2op.Agent import RecoPowerlinePerArea
import numpy as np
import pandas as pd
import os
import sys
import argparse
import pickle
import utils

try:
    from lightsim2grid import LightSimBackend
    bk_cls = LightSimBackend
except ImportError as exc:
    print(f"Error: {exc} when importing faster LightSimBackend")
    from grid2op.Backend import PandaPowerBackend
    bk_cls = PandaPowerBackend

# -------------------- UTILS --------------------------- #

def calc_sample_prob_weights(nof_each_type: np.ndarray) -> np.ndarray:
    """
    Calculate sample weights per type, depending on how many data points already exist of that type.
    Rare samples are weighted higher, common samples weighted lower with weights > 0 and <= 1.
    
    Input:
    nof_each_type: number of samples per type. E.g days of week, [1, 2, 0, 1, 3, 1, 0] means 1 monday, 2 tuesdays, etc.

    Output:
    sample_prob_per_type: probability of e.g saving a sample of each type next time a sample is drawn. 

    Example: 
    Input - [3, 1, 0, 2]

    Max value is 3. Probability of sampling each non-zero element is set to be proportional to max_val / input.
    Zero elements are set to 1. We get:
    [p, 3p, 1, 1.5p], where p is a normalization constant such that the elements containing p sum to 1, i.e
    p = 1 / 5.5.

    Output - [1 / 5.5, 3 / 5.5, 1,  1.5 / 5.5]
    """
    sample_prob_per_type = np.array(nof_each_type, dtype=float, copy=True) # copy array
    # get indices
    idx_zero = sample_prob_per_type == 0
    idx_not_zero = sample_prob_per_type != 0
    # handle case with only one non-zero elem
    if idx_not_zero.sum() == 1:
        sample_prob_per_type[idx_not_zero] = 0
        sample_prob_per_type[idx_zero] = 1
        return sample_prob_per_type
        
    max_val = sample_prob_per_type.max()
    # set probabilities for zero and non-zero elements separately
    sample_prob_per_type[idx_not_zero] = max_val / sample_prob_per_type[idx_not_zero]
    sample_prob_per_type[idx_not_zero] /= sample_prob_per_type[idx_not_zero].sum() # normalize
    sample_prob_per_type[idx_zero] = 1
    return sample_prob_per_type


def obs_to_df(obs, index=0) -> pd.DataFrame:
    """
    Convert observation to df.
    """
    d = obs.to_json()
    keys = []
    vals = []
    for k, v in d.items():
        keys.append(k)
        if len(v) == 1:
            vals.append(v[0])
        else:
            vals.append(np.array(v))
    return pd.DataFrame(data=[vals], columns=keys, index=[index])


# -------------------------------------------------------------- #



class DataCollector:
    """
    Used for collecting supervised training data for learning grid failure probabilities in case of contingencies.
    
    Two types of files are saved: 
    - observations (input data). Raw observations from grid
    - Y (learning target). Contains:
        - Binary grid survival per contingency. Values in [0, 1], len is alertable_line_ids in environment 
        - Failure timestep per contingency. Values in [-1, 0, 1, 2, ..., prediction_horizon] (-1 means no failure)\
            , len alertable_line_ids
        - If data is due to simulated contingencies, or a grid collapse encountered with no contingencies. Values \
            in [0, 1], len 1.

    The data needs to be preprocessed before being used for model training, see pre_process_data.ipynb.
   
    Methods:
    - run: run data collection
    """
    def __init__(self, env_instance, agent_instance, seed: int=0, prediction_horizon: int=12) -> None:
        """
        Input:
        - env_instance: 
        - agent_instance: 
        - seed: 
        - prediction_horizon: 
        """
        #self.probs_logging = 0.20
        self.nb_exp_to_save = 8000 # should be atleast 10x the size of the dim(s)
        self.prediction_horizon = prediction_horizon
        save_filename_template = "data/{type}_java_uni005_v4_{seed}.{format}"
        #save_filename_template = "data/{type}_test_{seed}.{format}"
        self.save_interval = 50
        
        self.obs_save_path = save_filename_template.format(type="obs", seed=seed, format="pickle")
        self.x_save_path = save_filename_template.format(type="X", seed=seed, format="csv")
        self.y_save_path = save_filename_template.format(type="Y", seed=seed, format="csv")

        self.env = env_instance

        self.alertable_line_ids = type(self.env.action_space).alertable_line_ids
        self.agent = agent_instance
        # Set seeds
        self.env.seed(seed)
        self.agent.seed(seed)
        self._rng = np.random.default_rng(seed)
        self.seed = seed
        np.random.seed(seed) # NOTE: perhaps this is unnecessary, and could harm reproducibility when multiprocessing?
    

    def _calc_sample_save_probability(self, df_X: pd.DataFrame, obs) -> float:
        return 0.05
        
        if len(df_X) == 0:
            return 0.05
        #n_each_dow = np.array([(df_X["day_of_week"] == i).sum() for i in range(0, 7)])
        #p_dow = calc_sample_prob_weights(n_each_dow)[obs.day_of_week]
        
        #n_each_hod = np.array([(df_X["hour_of_day"] == i).sum() for i in range(0, 24)])
        #p_hod = calc_sample_prob_weights(n_each_hod)[obs.hour_of_day]

        #n_each_dom = np.array([(df_X["day_of_month"] == i).sum() for i in range(1, 32)])
        #p_dom = calc_sample_prob_weights(n_each_dom)[obs.day]

        n_each_month = np.array([(df_X["month"] == i).sum() for i in range(1, 13)])
        p_month = calc_sample_prob_weights(n_each_month)[obs.month - 1]
        return p_month

    
    def _add_obs_to_buffer(self, buffer, obs):
        if len(buffer) == (self.prediction_horizon + 1):
            del buffer[0]
        elif len(buffer) > (self.prediction_horizon + 1):
            raise Warning("Buffer is larger than intended, something is wrong")
        buffer.append(obs.copy())
        return buffer

    def _y_sample2df(self, y_t: np.ndarray, exp_id: int) -> pd.DataFrame:
        columns = [f"survival_cont_{i}" for i in self.alertable_line_ids] + \
            [f"fail_t_cont_{i}" for i in self.alertable_line_ids] + ["sim"]
        return pd.DataFrame(data=[y_t], columns=columns, index=[exp_id])


    def _gather_df_y(self, exp_id: int) -> pd.DataFrame:
        """
        Apply one contingency at a time, simulate with policy for self.prediction_horizon steps. If failure, set survival[i] = 0, else 1,
        where i is the idx of the contingency. Save timestep for failure in failure_timestep[i]. Also store that data is simulated.
        y = [survival, failure_timestep, simulated].
        Input
        exp_id: used to index df
        Output
        df_y_t: dataframe with 1 row, containing y-data
        """
        survival = np.zeros(len(self.alertable_line_ids)) # per contingency
        failure_timesteps = np.zeros(len(self.alertable_line_ids)) # per contingency
        for cont_id, line_id in enumerate(self.alertable_line_ids):
            env_cpy = self.env.copy()
            obs_cpy, r_cpy, done_cpy, info_cpy = env_cpy.step(env_cpy.action_space({"set_line_status": [(line_id, -1)]}))
            env_steps = 0 # NOTE: dont replace this by the index from the for-loop, behaviour is not equivalent!
            for _ in range(self.prediction_horizon): 
                if done_cpy:
                    # Break before step below, env already done
                    break
                # Prevent the reconnection of the powerline by the agent. 
                # Since agent might use time_before_cooldown_line when making a decision,
                # we decided to simulate that it will be able to reconnect after self.prediction_horizon steps after attack. 
                # There are probably other ways to do this, but shouldnt matter too much hopefully.
                obs_cpy.time_before_cooldown_line[line_id] = self.prediction_horizon - env_steps
                obs_cpy, r_cpy, done_cpy, info_cpy = env_cpy.step(self.agent.act(obs_cpy, r_cpy)) # Step
                env_steps += 1 # env_steps must be updated after the break call above
            survival[cont_id] = (env_steps == self.prediction_horizon) and (not done_cpy)
            failure_timesteps[cont_id] = env_steps if done_cpy else -1
        y_t = np.concatenate([survival, failure_timesteps, [True]])
        df_y_t = self._y_sample2df(y_t, exp_id)
        return df_y_t


    def _episode_helper(self, obs_list, df_X, df_x_t, df_Y, df_y_t, exp_id):
        """
        Concats data, increments exp_id, prints progress and saves data.
        """
        df_X = pd.concat([df_X, df_x_t])
        df_Y = pd.concat([df_Y, df_y_t])

        exp_id += 1
        print(f"Sample {exp_id} / {self.nb_exp_to_save}")

        if exp_id % self.save_interval == 0 and exp_id > 0:
            with open(self.obs_save_path, "wb") as file:
                pickle.dump(obs_list, file)
            #df_X.to_csv(self.x_save_path, sep=",")
            df_Y.to_csv(self.y_save_path, sep=",")
        
        return df_X, df_Y, exp_id



    def run(self):
        """
        Gather and save data from env using agent
        """
        df_X = pd.DataFrame() # input data
        df_Y = pd.DataFrame() # labels
        observations = []
        exp_id = 0
        # rng = np.random.default_rng() # random number generator
        # run the experiments
        ep = 0
        while True:
            # Start new episode
            chronic_id = self.env.chronics_handler.sample_next_chronics() # see starting kit comp nb 2
            obs = self.env.reset()
            done = False
            reward = self.env.reward_range[0]
            self.agent.reset(obs)
            buffer = [obs.copy()] # made for storing the last horizon + 1 obs this episode
            print(f"\nChronic id, episode: {chronic_id}, {ep}\n")
            while True: # step through chronic:
                obs, reward, done, info = self.env.step(self.agent.act(obs, reward, done))
                buffer = self._add_obs_to_buffer(buffer, obs)

                # In case of grid failure
                if len(info["exception"]) > 0 and len(buffer) == (self.prediction_horizon + 1):
                    # We add a y-vector of 0s instead of simulating contingencies.
                    # the corresponding x-vector is extracted from first obs in buffer
                    observations.append(buffer[0])
                    df_x_t = obs_to_df(buffer[0], exp_id)
                    survival = [0 for i in range(len(self.alertable_line_ids))]
                    failure_timestep = [self.prediction_horizon for i in range(len(self.alertable_line_ids))]
                    y_t = np.concatenate([survival, failure_timestep, [False]])
                    df_y_t = self._y_sample2df(y_t, exp_id)
                    df_X, df_Y, exp_id = self._episode_helper(
                        obs_list=observations, 
                        df_X=df_X, 
                        df_x_t=df_x_t, 
                        df_Y=df_Y, 
                        df_y_t=df_y_t, 
                        exp_id=exp_id)
                
                if done:
                    ep += 1 
                    break

                # do I save the "state" for my X,Y supervised problem there
                do_i_log = self._rng.uniform() <= self._calc_sample_save_probability(df_X, obs) #0.05 
                if do_i_log:
                    if obs.current_step >= obs.max_step - self.prediction_horizon - 1:
                        break # dont sample if simulations will reach or pass the end of the chronics
                    observations.append(obs)
                    df_x_t = obs_to_df(obs, exp_id)
                    df_y_t = self._gather_df_y(exp_id)
                    df_X, df_Y, exp_id = self._episode_helper(
                        obs_list=observations, 
                        df_X=df_X, 
                        df_x_t=df_x_t, 
                        df_Y=df_Y, 
                        df_y_t=df_y_t, 
                        exp_id=exp_id)

                    if exp_id >= self.nb_exp_to_save:
                        # stop the current episode
                        break
                        
            if exp_id >= self.nb_exp_to_save:
                # stop gathering experiments
                break
        

def main(seed):
    no_opp_kwargs = grid2op.Opponent.get_kwargs_no_opponent()
    env = grid2op.make(
        "l2rpn_idf_2023",
        **no_opp_kwargs, backend=bk_cls())

    agent = utils.load_agent_from_submission_file(env, "agent_FinalSubmission_javaness")

    """agent = RecoPowerlinePerArea(
        env.action_space, 
        env._game_rules.legal_action.substations_id_by_area)"""

    data_extr = DataCollector(env, agent, seed=seed)
    data_extr.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="Seed to use for run", type=int)
    args = parser.parse_args()
    main(int(args.seed))

    #main(99)
