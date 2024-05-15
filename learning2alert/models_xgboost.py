# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib
#matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
import utils
import numpy as np
import time
from sklearn import metrics
import plotly.express as px
import plotly.io as pio
from sklearn.base import BaseEstimator
import joblib
import pickle
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import datetime
from sklearn import metrics
import datetime
import os
from typing import Union

pio.renderers.default = "browser"


class MultiWeightXGBWrapper(BaseEstimator):
    """
    Creates a XGB model with multidimensional output, with different scale_pos_weights for each model.
    """
    def __init__(self, model_constructor, weights, kwargs=None) -> None:
        self.weights = weights
        self.model_constructor = model_constructor
        self.kwargs = kwargs
        self.models = self._generate_models()

    def _generate_models(self):
        models = []
        for w in self.weights:
            models.append(self.model_constructor(scale_pos_weight=w, **self.kwargs))
        return models

    def fit(self, X: Union[np.ndarray, pd.DataFrame], Y: Union[np.ndarray, pd.DataFrame]):
        try:
            X = X.values
            Y = Y.values # for pd.DataFrame, get values
        except AttributeError:
            pass
        for i, model in enumerate(self.models):
            model.fit(X, Y[:, i])
            
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        Y_pred = []
        for i, model in enumerate(self.models):
            Y_pred.append(model.predict(X))
            #Y_pred.append(model.predict(X))
        return np.array(Y_pred).T
    

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the probabilities that X does not belong to (label==0) AND belongs to (label==1) each class.
        Intended for multilabel classification.

        Input
        - X: input data, shape (n_samples, n_features)

        Output
        - P: List of probabilities of X corresponding to label 0 and 1. List has length n_classes,
             each element is array with shape (n_samples, 2).
        """
        p_pred = []
        for model in self.models:
            p_pred.append(model.predict_proba(X))
        return np.array(p_pred)


    def predict_proba_ones(self, X: pd.DataFrame):
        """
        Predict the probabilities that X belongs to each class (label==1).
        Intended for multilabel classification.

        Input
        - X: input data, shape (n_samples, n_features)

        Output
        - P: probabilities, shape (n_samples, n_classes)
        """
        p_pred = []
        for model in self.models:
            p_pred.append(model.predict_proba(X)[:, 1])
        return np.array(p_pred).T
        
    def get_feature_importance(self):
        feature_importance_mean = 0
        for model in self.models:
            feature_importance_mean += model.feature_importances_
        return feature_importance_mean / len(self.models)

    def classes_(self):
        if self.models:
            classes = []
            for model in self.models:
                classes.append(type(model))
            return classes

    def set_params(self, **params):
        if not params:
            return self
        if params.get("weights"):
            self.weights = params.pop("weights")
        if params.get("model_constructor"):
            self.model_constructor = params.pop("model_constructor")
        # Remaining params belong to kwargs. Add new, update old vals
        self.kwargs.update(params)
        self.models = self._generate_models() # create new models
        return self

    def get_params(self, deep=True):
        return {
            "kwargs": self.kwargs,
            "model_constructor": self.model_constructor,
            "weights": self.weights,
        }


class Trainer():
    """
    Train, tune and eval a xgboost model.

    Methods:
    - train: train and save model checkpoint. Also save classification thresholds.
    - tune: hyperparameter search, save best parameters
    - eval: Load model and optionally thresholds, evaluate on test set.
    """
    def __init__(
        self, 
        load_dir,
        seed: int=99,
        ) -> None:
        """
        Load data, calculate class weights.

        Input:
        - load_dir: path to directory containing files "X_train.csv", "X_test.csv", "Y_train.csv", "Y_test.csv"
        - seed: random seed to use for rng
        """

        self.load_dir = load_dir
        self.seed = seed
        self.data = utils.load_data(
            self.load_dir,
            val=True,
            seed=self.seed,
        )
        self.class_weights = 1 - utils.calc_fraction_ones_per_col(self.data["Y_train"])


    def _create_logdir(self, experiment_name):
        logdir = f"experiments/xgboost/{experiment_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        utils.make_dir_if_doesnt_exist(logdir)
        return logdir


    def _get_balanced_accuracy_multiclass_scorer(self, kwargs):
        return metrics.make_scorer(
            utils.balanced_accuracy_multiclass,
            greater_is_better=True,
            **kwargs,
        )


    def train(
        self, 
        experiment_name: str="default",
        model_kwargs: dict=None,
        ):
        """
        Train and save regression model, save thresholds for classification.

        Input:
        - experiment_name: Used to generate folder where model checkpoint, classification threshold, results are saved.
        - model_kwargs: kwargs passed to XGBRegressor init
        """
        self.model_kwargs = { # default values
            "n_estimators": 100, # 2, 6, 100
            "max_depth": 100, # 2, 12, 10
            "gamma": 1,
            "min_child_weight": 1, # 100 is fine
            "seed": 35,
        }
        model_constructor = XGBRegressor
        self.model_kwargs["objective"] = 'reg:logistic'
        if model_kwargs:
            self.model_kwargs.update(model_kwargs)

        logdir = self._create_logdir(experiment_name)
        cp_path = logdir + "/checkpoint.pkl"

        model = MultiWeightXGBWrapper(model_constructor, self.class_weights, self.model_kwargs)

        print("\nTraining model...\n")
        model.fit(self.data["X_train"].values, self.data["Y_train"].values)
        print("\nDone\n")

        # Calculate & save thresholds
        P_pred_val = model.predict(self.data["X_val"].values)
        selected_thresholds = utils.select_thresholds(self.data["Y_val"], P_pred_val, plot=False)
        np.save(logdir + "/thresholds.npy", selected_thresholds)

        utils.pickle_model(model, cp_path)
        config = {
            "load_dir": self.load_dir,
            "seed": self.seed,
            "model_kwargs": self.model_kwargs
        }
        utils.save_params(config, logdir + "/config.yml")
        return logdir

        
    def tune(
        self, 
        experiment_name: str="default",
        tune_params: dict=None,
        ):
        """
        Run hyperparameter search.

        Input:
        - experiment_name: Used to generate folder where best model checkpoint and parameters are saved.
        - tune_params: dict with key XGBRegressor argument name, value list of values 
        """
        self.tune_params = {
            "max_depth": [1],
            #"max_depth": [1, 10, 100], # np.logspace(0, 1.5, 20),
            #"n_estimators": [1, 10, 100], #np.logspace(0, 2.5, 20),
            #"gamma": [0, 1, 10, 100, 1000], # [0] + list(np.logspace(-3, 3, 20)),
            #"min_child_weight": [0, 1, 10, 100, 1000]  # np.logspace(-3, 3, 20),
        }
        if tune_params is not None:
            self.tune_params.update(tune_params)

        logdir = self._create_logdir(experiment_name)
        cp_path = logdir + "/checkpoint.pkl"
        kwargs = {}
        model_constructor = XGBRegressor
        kwargs["objective"] = 'reg:logistic'
        scoring = self._get_balanced_accuracy_multiclass_scorer({"weights": [0.33, 0.67]})
        model = MultiWeightXGBWrapper(model_constructor, self.class_weights, kwargs)
        
        #clf = RandomizedSearchCV(model, tune_params, scoring=scoring, n_iter=30)
        clf = GridSearchCV(model, self.tune_params, scoring=scoring)

        print("\nTuning model...\n")
        t0 = time.time()

        # NOTE: tune got stuck before, so added .values to x and y. TODO: Run tune
        
        search = clf.fit(self.data["X_train"].values, self.data["Y_train"].values)
        print(f"Time: {datetime.timedelta(seconds=time.time()-t0)}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_}")

        utils.pickle_model(search.best_estimator_, cp_path)
        # Store params
        config = {
            "load_dir": self.load_dir,
            "seed": self.seed,
            "model_kwargs": kwargs,
            "best_params": search.best_params_,
            "tune_params": self.tune_params,
        }
        utils.save_params(config, logdir + "/config.yml")
        return cp_path
        


    def eval(self, logdir, plot: bool=False):
        """
        Load and eval model. Loads threshold file if available, otherwise uses 0.5.
        """
        print("\nEvaluating model...\n")
        # Load model
        model = utils.load_pickled_model(os.path.join(logdir, "checkpoint.pkl"))
        # Load thresholds
        threshold_path = os.path.join(logdir, "thresholds.npy")
        try:
            print(f"\nLoading threshold from:\n{threshold_path}\n")
            selected_thresholds = np.load(threshold_path)
        except OSError:
            print(f"\nCould not find threshold at path:\n{threshold_path}\nSetting thresholds = 0.5\n")
            selected_thresholds = 0.5

        # Evaluate on test set
        P_pred_test = model.predict(self.data["X_test"].values)
        Y_pred_test = (P_pred_test > selected_thresholds) * 1 # convert prob to label pred
    
        conf_df_per_col = utils.calc_confusion_metrics_per_col(self.data["Y_test"], Y_pred_test)
        conf_df = utils.calc_confusion_metrics(self.data["Y_test"], Y_pred_test)
        results_df = pd.DataFrame(
            data=[[
                conf_df_per_col['true_pos'].min(), 
                conf_df["true_pos"][0], 
                conf_df_per_col['true_pos'].max(), 
                conf_df_per_col['true_neg'].min(),
                conf_df["true_neg"][0],
                conf_df_per_col['true_neg'].max(),
            ]],
            columns=["tp_min", "tp_avg", "tp_max", "tn_min", "tn_avg", "tn_max"],
        ).round(4)
        score_balanced = utils.balanced_accuracy_multiclass(
            self.data["Y_test"], 
            Y_pred_test, 
            weights=[0.33, 0.67])

        print(f"Selected thresholds: {selected_thresholds}")
        print(conf_df_per_col)
        print(results_df)
        print(f"Balanced accuracy score weighted: {score_balanced}")

        results_path = os.path.join(logdir, "results.csv")
        if not os.path.isfile(results_path):
            results_df.to_csv(results_path, sep=",")
            print(f"\nSaving results to: {results_path}\n")

        if plot:
            fig = px.bar(x=self.data["X_val"].columns, y=model.get_feature_importance(), title="Mean feature importance")
            fig.write_html("figs/feature_importance.html")
            fig.show()


def main():
    trainer = Trainer(
        "data/java_minimal",
    )
    """cp = trainer.train(
        experiment_name="java_minimal",
        model_kwargs={
            #"n_estimators": 1, 
            #"max_depth": 2,
            #"gamma": 1,
            #"min_child_weight": 1, 
        }
    )"""
    """cp = trainer.tune(
        experiment_name="java_minimal_tune",
        tune_params = {
            "max_depth": [1, 2],
            "n_estimators": [1],
            #"max_depth": [1, 10, 100], # np.logspace(0, 1.5, 20),
            #"n_estimators": [1, 10, 100], #np.logspace(0, 2.5, 20),
            #"gamma": [0, 1, 10, 100, 1000], # [0] + list(np.logspace(-3, 3, 20)),
            #"min_child_weight": [0, 1, 10, 100, 1000]  # np.logspace(-3, 3, 20),
        },
    )"""
    #cp = "experiments/xgboost/java_minimal/20231127-174459"
    cp = "experiments/xgboost/java_minimal/20231127-174855/"
    trainer.eval(cp, plot=False)
    

if __name__ == "__main__":
    main()
