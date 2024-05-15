# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.Agent import BaseAgent
from grid2op import Agent
import state_extractor

class TrainedAlertModule(BaseAgent):
    """
    Wrapper around a grid2op agent, adding an alert action based on grid failure probability predicted by survival model.
    """
    def __init__(
        self, 
        action_space: ActionSpace, 
        instantiated_controller: Agent,
        instantiated_survival_model,  
        instantiated_state_extractor: state_extractor.StateExtractor,
        alert_threshold=0.5):
        """
        Input:
        - action_space: action space of environment
        - instantiated_controller: the agent that will operate the grid.
        - instantiated_survival_model: a model which predicts survival probability per alertable line. The model \
            should already be trained on data gathered by 'instantiated_controller' class.
        - instantiated_state_extractor: the state extractor used when training the model. Makes sure model gets correct input.
        """
        super().__init__(action_space)
        self._controller = instantiated_controller
        self._alert_threshold = alert_threshold
        self._survival_model = instantiated_survival_model
        self._state_extractor = instantiated_state_extractor
        self._keras_model = "keras" in str(type(self._survival_model))
        

    def act(self, observation: BaseObservation, reward: float, done: bool = False):
        """
        Add raise_alert to grid controller action.
        """
        action = self._controller.act(observation, reward, done)
        # determine alert action:
        state = self._state_extractor.extract_state(observation, exp_id=0)
        if self._keras_model:
            success_prob = self._survival_model(state)
        else:
            success_prob = self._survival_model.predict(state)
        idx = np.where(success_prob <= self._alert_threshold)[1] # failure_prob shape 1xN
        action.raise_alert = [i for i in idx]
        return action
