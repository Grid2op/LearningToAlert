# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

import os
import sys
import numpy as np
import grid2op
from grid2op.Runner import Runner
import plotly.io as pio
import tensorflow as tf
from grid2op.Reward import BaseReward, _AlertTrustScore, AlertReward
from grid2op.dtypes import dt_float
from grid2op.Agent import AlertAgent, RecoPowerlinePerArea, DoNothingAgent
import state_extractor
from agent import TrainedAlertModule
import utils
from typing import Literal, List
import json
from grid2op.dtypes import dt_int
import datetime
from grid2op.Episode import EpisodeData
import pandas as pd
from grid2op.utils.l2rpn_idf_2023_scores import ScoreL2RPN2023
from grid2op.Reward import BaseReward, RedispReward, L2RPNWCCI2022ScoreFun
from tqdm import tqdm
from models_xgboost import MultiWeightXGBWrapper
from grid2op import Agent

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Backend class to use
try:
    from lightsim2grid.lightSimBackend import LightSimBackend
    BACKEND = LightSimBackend
except ModuleNotFoundError:
    from grid2op.Backend import PandaPowerBackend
    BACKEND = PandaPowerBackend
print("you have loaded "+str(BACKEND))


class ConfidenceRateReward(BaseReward):
    """
    This will give the confidence rate, hence the percentage of alerts not raised
    """
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)

    def initialize(self, env: "grid2op.Environment.BaseEnv"):
        self.n_alertable_lines = len(env.alertable_line_ids)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            res = 1-len(np.where(action.raise_alert)[0])/self.n_alertable_lines 
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        # print(f"\t env.backend.get_line_flow(): {env.backend.get_line_flow()}")
        return res


class EvalRunner:
    """
    Pass agent to grid2op.Runner.run, optionally adding an alert module. Seeds can be manually specified or loaded from file.

    Methods:
    - run: start evaluation by calling runner
    """
    def __init__(
        self,
        agent_instance,
        add_alert_module: bool,
        agent_save_name: str,
        results_save_dir="stored_agents/default",
        nb_episodes=10, # -1 to run all seeds
        env_name="input_data_test",
        reward_class=None,
        alert_module_kwargs=None,
        episode_config_path: str=None,
        agent_seeds: List[int]=None,
        env_seeds: List[int]=None,
        episode_ids: List[int]=None,
        max_iter: int=None,
        ) -> None:
        """
        Input:
        - agent_instance: agent to be used for operating on grid
        - add_alert_module: if an alert module should be added to agent
        - agent_save_name: used with results_save_dir to create a save path for results
        - results_save_dir: used with agent_save_name to create a save path for results
        - nb_episodes: number of episodes to run
        - env_name: name of grid2op environment
        - reward_class: reward used for evaluating run. If none, uses _AlertTrustScore
        - alert_module_kwargs: kwargs for initializing alert module:
            - model_load_path: path to where model checkpoint is located
            - state_extractor_class: state extractor used when gathering training data for model. \
                Default: state_extractor.RhoMaintenanceDatesPower
            - state_extractor_kwargs: kwargs to pass to state_extractor_class at init
            - nn_model: if path is keras model (True) or xgboost model (False)
            - alert_threshold: survival probability <= alert_threshold leads to raising alarms. \
                If None, loads from model_load_path/thresholds.npy
        - episode_config_path: path to score config where seeds are loaded from. Either specify this or specify seeds manually.
        - agent_seeds: manually specify agent seeds (instead of loading from episode_config_path). 
        - env_seeds: manually specify environment seeds (instead of loading from episode_config_path). 
        - episode_ids: manually specify episode ids (instead of loading from episode_config_path). 
        - max_iter: max number of iterations per episode.
        """

        self.agent_instance = agent_instance
        self.add_alert_module = add_alert_module
        self.nb_episodes = nb_episodes
        self.env_name = env_name
        self.reward_class = reward_class

        if reward_class is None:
            self.reward_class = _AlertTrustScore # default value

        self.alert_module_kwargs = {  # default values
            "model_load_path": "path_where_checkpoint_and_thresholds_is_saved",
            "state_extractor_class": state_extractor.RhoMaintenanceDatesPower, # model should have been trained using this StateExtractor
            "state_extractor_kwargs": {},
            "nn_model": True, # neural network or xgboost model
            "alert_threshold": None, # None means threshold will be loaded from model_load_path/thresholds.npy
        }
        if alert_module_kwargs is not None:
            self.alert_module_kwargs.update(alert_module_kwargs) # update

        self.save_path = os.path.join(
            results_save_dir, 
            agent_save_name,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.max_iter = max_iter # 2016 is max ep length in l2rpn_2023
        self.episode_config_path = episode_config_path
        self._env = grid2op.make(self.env_name, test=False, backend=BACKEND(), reward_class=self.reward_class,
            other_rewards={"confidence_rate": ConfidenceRateReward})

        if self.episode_config_path: # user specified episodes and seeds via config file
            assert agent_seeds is None and env_seeds is None and episode_ids is None, \
                "Can't specify 'agent_seeds', 'env_seeds' or 'episode_ids' if 'episode_config_path' is specified"

            with open(self.episode_config_path) as file:
                ep_config = json.load(file)
            nb_run = int(ep_config["nb_scenario"])

            assert self.nb_episodes <= nb_run, f"Specified 'nb_episodes' {self.nb_episodes}, but only {nb_run} exist in ep_config {self.episode_config_path}"
            
            if self.nb_episodes < 1:
                self.nb_episodes = nb_run

            np.random.seed(int(ep_config["score_config"]["seed"]))
            max_int = np.iinfo(dt_int).max
            # env seeds and episode ids are read from the json
            env_seeds = []
            episode_ids = []
            for el in sorted(self._env.chronics_handler.real_data.subpaths):
                env_seeds.append(int(ep_config["episodes_info"][os.path.split(el)[-1]]["seed"]))
                episode_ids.append(os.path.split(el)[-1])
            # agent seeds are generated with the provided random seed
            agent_seeds = list(np.random.randint(max_int, size=nb_run))
            self.env_seeds = env_seeds[:self.nb_episodes]
            self.agent_seeds = agent_seeds[:self.nb_episodes]
            self.episode_ids = episode_ids[:self.nb_episodes]
        else: # user specified seeds and episodes directly
            if self.nb_episodes < 1:
                self.nb_episodes = len(env_seeds)

            self.env_seeds = env_seeds[:nb_episodes] if env_seeds else env_seeds
            self.agent_seeds = agent_seeds[:nb_episodes] if agent_seeds else agent_seeds
            self.episode_ids = episode_ids[:nb_episodes] if episode_ids else episode_ids
                    

    def _load_trained_alert_module(self, controller: Agent) -> TrainedAlertModule:
        """
        Initialize TrainedAlertModule with specified instantiated controller and config settings.
        """
        state_extractor = self.alert_module_kwargs["state_extractor_class"](
            self._env,
            **self.alert_module_kwargs["state_extractor_kwargs"]
            )
        # Load prediction model and modify agent_name based on settings
        if self.alert_module_kwargs["nn_model"]:
            checkpoint_path = os.path.join(self.alert_module_kwargs["model_load_path"], "checkpoint")
            print(f"\nLoading model from file: {checkpoint_path}\n")
            survival_model = tf.keras.models.load_model(checkpoint_path, compile=False)
        else:
            checkpoint_path = os.path.join(self.alert_module_kwargs["model_load_path"], "checkpoint.pkl")
            print(f"\nLoading model from file: {checkpoint_path}\n")
            survival_model = utils.load_pickled_model(checkpoint_path)
        
        threshold = self.alert_module_kwargs["alert_threshold"]
        if threshold is None:
            threshold_path = os.path.join(self.alert_module_kwargs["model_load_path"], "thresholds.npy")
            print(f"\nLoading threshold from file: {threshold_path}\n")
            threshold = np.load(threshold_path)

        # Make and return agent
        return TrainedAlertModule(
            self._env.action_space, 
            instantiated_controller=controller,
            instantiated_state_extractor=state_extractor,
            instantiated_survival_model=survival_model,
            alert_threshold=threshold,
        )

    def _save_config(self, savedir):
        cfg = vars(self)
        utils.save_params(cfg, savedir + "/config.yml")
            
    def run(self):
        """
        Call grid2op runner. Optionally add alert module to agent_instance
        """
        utils.make_dir_if_doesnt_exist(self.save_path)
        self._save_config(self.save_path)

        if self.add_alert_module:
            self.agent_instance = self._load_trained_alert_module(self.agent_instance)
        runner = Runner(**self._env.get_params_for_runner(),
                        agentInstance=self.agent_instance,
                        agentClass=None)
        res = runner.run(nb_episode=self.nb_episodes, nb_process=1, path_save=self.save_path, 
            agent_seeds=self.agent_seeds, env_seeds=self.env_seeds, max_iter=self.max_iter, 
            episode_id=self.episode_ids, add_detailed_output=True)

        cum_rewards = []
        nb_time_steps = []
        for elem in res:
            id_chron, name_chron, cum_reward, nb_time_step, max_ts, episode_data = elem
            cum_rewards.append(cum_reward)
            nb_time_steps.append(nb_time_step)
        results_df = pd.DataFrame(
            data=np.array([cum_rewards, nb_time_steps]).T,
            index=self.episode_ids,
            columns=["reward", "n_steps"],
        )
        results_df.to_csv(os.path.join(self.save_path, "results.csv"), sep=",")


def main():
    agent_save_name = "alert_module_javaness"
    # Pick 1 of 3 settings based on agent_save_name:
    env_name = "input_data_test"
    env = grid2op.make(env_name)
    if agent_save_name == "alert_module_javaness":
        agent_instance = utils.load_agent_from_submission_file(env, "agent_FinalSubmission_javaness")
        add_alert_module = True
        agent_instance.alert_module = DoNothingAgent(env.action_space) # Replace agents alert_module with do nothing
    elif agent_save_name == "javaness_agent":
        agent_instance = utils.load_agent_from_submission_file(env, "agent_FinalSubmission_javaness")
        add_alert_module = False
    elif agent_save_name == "javaness_agent_mod":
        agent_instance = utils.load_agent_from_submission_file(env, "agent_FinalSubmission_javaness_mod")
        add_alert_module = False

    config = {
        "agent_instance": agent_instance,
        "add_alert_module": add_alert_module,
        "agent_save_name": agent_save_name,
        "results_save_dir": "stored_agents/test",
        "nb_episodes": -1, # 208
        "env_name": env_name,
        "reward_class": None,
        "episode_config_path": "config_test.json",
        "agent_seeds": None, # [304680494],# [837303025],  #[105069337], # [1536643794], # [304680494],
        "env_seeds": None, # [552965536],#[2070057260], # [120582381], # [1998202551], #[552965536],
        "episode_ids": None, #["2035-12-10_10"],# ["2035-09-10_19"], # ["2035-03-19_0"], # ["2035-06-11_7"], # ["2035-12-10_10"],
        "max_iter": None,
        "alert_module_kwargs": {
            "state_extractor_class": state_extractor.RhoMaintenanceDatesPower,
            "model_load_path": "experiments/java_v3/20231204-174630_99",
            "nn_model": True,
            "alert_threshold": 0.5,
            },
    }
    EvalRunner(**config).run()


if __name__ == "__main__":
    main()
