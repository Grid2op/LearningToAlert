# Copyright (c) 2023-2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE)
# This file is part of SL Alert Agent, an agent module that learns to send alerts through supervised learning when the agent is prone to failure.

import tensorflow as tf
import numpy as np
import copy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import pandas as pd
import datetime
import utils
import utils
import pre_process_data_utils
import os
from typing import List, Dict
from sklearn.model_selection import train_test_split
import argparse

#tf.config.run_functions_eagerly(True) # Allow debugging of functions with decorator @tf.function !!

# aliases because pylance doesnt understand tf imports: 
layers = tf.keras.layers
regularizers = tf.keras.regularizers
losses = tf.keras.losses
optimizers = tf.keras.optimizers
metrics = tf.keras.metrics
callbacks = tf.keras.callbacks


class MSEWeighted:
    """
    MSE weighted based on fraction_ones. Assumes y_pred only contains 0:s and 1:s.
    """
    def __init__(self, fraction_ones: np.array, zeros_mult_factor: float=1.) -> None:
        self.fraction_ones = fraction_ones
        self.zeros_mult_factor = zeros_mult_factor
    
    # @tf.function
    def mse_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        sample_weights = utils.calc_sample_weights(
            y_true, 
            self.fraction_ones, 
            zeros_mult_factor=self.zeros_mult_factor,
        )
        return K.mean(K.square(y_true - y_pred) * sample_weights) # / K.sum(sample_weights)


class BinaryCrossEntropyWeighted:
    """
    Weighted binary cross entropy
    """
    def __init__(self, fraction_ones: np.array) -> None:
        self.fraction_ones = fraction_ones

    # @tf.function
    def bce_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        sample_weights = utils.calc_sample_weights(y_true, self.fraction_ones)
        # TODO: test it, consider if there are other numerical stability concerns
        term0 = -y_true * K.log(y_pred + K.epsilon())
        term1 = -(1 - y_true) * K.log(1 - y_pred + K.epsilon())
        return K.mean((term0 + term1) * sample_weights) * 2 # *2 to match keras bce with uniform weights



class Specificity(tf.keras.metrics.Metric):
    """
    Calculate true negative rate
    """
    def __init__(self, name="specificity", threshold=0.5, **kwargs):
        super().__init__(name, **kwargs)
        self.true_negatives = self.add_weight(name='true_neg', initializer='zeros')
        self.negatives = self.add_weight(name='neg', initializer='zeros')
        self.threshold = threshold

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        zeros_at = tf.cast(y_true == 0, tf.float32)
        y_pred_class = tf.cast((y_pred > self.threshold), tf.float32)
        correct_pred = tf.cast((y_pred_class == 0), tf.float32)
        true_neg = K.sum(correct_pred * zeros_at) 
        neg = K.sum(zeros_at)
        self.true_negatives.assign_add(true_neg)
        self.negatives.assign_add(neg)

    def result(self):
        return self.true_negatives / self.negatives
    
    def reset_state(self):
        self.true_negatives.assign(0)
        self.negatives.assign(0)

#@tf.keras.saving.register_keras_serializable()
class MinSpecificity(tf.keras.metrics.Metric):
    """
    Calculate true negative rate per column, return min value
    """
    def __init__(self, n_cols, name="min_specificity", threshold=0.5, **kwargs):
        super().__init__(name, **kwargs)
        self.n_true = self.add_weight(name='n_true', initializer='zeros', shape=n_cols)
        self.n_tot = self.add_weight(name='n_tot', initializer='zeros', shape=n_cols)
        self.threshold = threshold
        self.n_cols = n_cols

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        zeros_at = tf.cast(y_true == 0, tf.float32)
        y_pred_class = tf.cast((y_pred > self.threshold), tf.float32)
        correct_pred = tf.cast((y_pred_class == 0), tf.float32)
        true_neg = K.sum(correct_pred * zeros_at, axis=0) 
        neg = K.sum(zeros_at, axis=0)
        self.n_true.assign_add(true_neg)
        self.n_tot.assign_add(neg)

    def result(self):
        return K.min(self.n_true / self.n_tot)
    
    def reset_state(self):
        self.n_true.assign(tf.zeros(self.n_cols))
        self.n_tot.assign(tf.zeros(self.n_cols))

    def get_config(self):
        """Returns the serializable config of the metric."""
        return {"name": self.name, "dtype": self.dtype, "n_cols": self.n_cols, "threshold": self.threshold}


class MaxSpecificity(MinSpecificity):
    """
    Calculate true negative rate per column, return max value
    """
    def __init__(self, n_cols, name="max_specificity", threshold=0.5, **kwargs):
        super().__init__(n_cols, name, threshold, **kwargs)

    def result(self):
        return K.max(self.n_true / self.n_tot)


class MinRecall(MinSpecificity):
    """
    Calculate true positive rate per column, return min value
    """
    def __init__(self, n_cols, name="min_recall", threshold=0.5, **kwargs):
        super().__init__(n_cols, name, threshold, **kwargs)
        
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        ones_at = tf.cast(y_true == 1, tf.float32)
        y_pred_class = tf.cast((y_pred > self.threshold), tf.float32)
        correct_pred = tf.cast((y_pred_class == 1), tf.float32)
        true_neg = K.sum(correct_pred * ones_at, axis=0) 
        neg = K.sum(ones_at, axis=0)
        self.n_true.assign_add(true_neg)
        self.n_tot.assign_add(neg)


class MaxRecall(MinRecall):
    """
    Calculate true positive rate per column, return max value
    """
    def __init__(self, n_cols, name="max_recall", threshold=0.5, **kwargs):
        super().__init__(n_cols, name, threshold, **kwargs)

    def result(self):
        return K.max(self.n_true / self.n_tot)


def create_fc_model(
    input_dim, 
    hidden_dims, 
    output_dim, 
    norm_layer=None, 
    use_batch_norm=False,
    dropout_rate=None,
    l2=1e-6,
    name="fc_model",
    ):
    """
    Create and compile keras model.

    If both batch norm and dropout is enabled, dropout is only applied right before the output later.
    """
    # NOTE: Drop out should only be used after all of the BN layers
    # Dropout causes inconcistency in the variance for the next BN layer! 

    if hidden_dims is None:
        hidden_dims = []

    # Create model add and input layer
    model = tf.keras.models.Sequential(name=name)
    model.add(tf.keras.Input((input_dim,)))
    if norm_layer:
        model.add(norm_layer)
 
    # Add hidden layers:
    for i, h_dim in enumerate(hidden_dims):
        model.add(layers.Dense(h_dim, kernel_regularizer=regularizers.L2(l2)))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        if not use_batch_norm and dropout_rate is not None:
            model.add(layers.Dropout(dropout_rate))
    
    if use_batch_norm and dropout_rate is not None:
        model.add(layers.Dropout(dropout_rate))
    
    # Add output layer:
    model.add(layers.Dense(output_dim, activation="sigmoid")) # each elem in output vector should be a probability
    return model



def resnet_block(x_in, hidden_dim, l2):
    """
    Resnet block with 2 layers and a skip connection.
    """
    x = layers.Dense(hidden_dim, kernel_regularizer=regularizers.L2(l2))(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(hidden_dim, kernel_regularizer=regularizers.L2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x_in, x]) # skip connection
    x = layers.Activation("relu")(x)
    return x


def create_resnet_model(
    input_dim: int, 
    hidden_dim: int,
    output_dim: int, 
    n_res_blocks: int,
    norm_layer: layers.Normalization=None, 
    dropout_rate: int=None,
    l2=0.01,
    name="resnet_model"):
    """
    Compile a resnet model.
    """
    # input block
    x_in = tf.keras.Input((input_dim,))
    if norm_layer:
        x = norm_layer(x_in)
    x = layers.Dense(hidden_dim, kernel_regularizer=regularizers.L2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # resnet blocks:
    for i in range(n_res_blocks):
        x = resnet_block(x, hidden_dim, l2)

    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # output block:
    out = layers.Dense(output_dim, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=x_in, outputs=out, name=name)
    return model


class Trainer():
    """
    Train and evaluate a keras model.

    Methods:
    - train: train and save model checkpoint. Also save classification thresholds.
    - eval: Load model and optionally thresholds, evaluate on test set.
    """
    def __init__(
        self,
        data_dir: str,
        seed: int=99,
        weighted_mse_kwargs: Dict=None,
        #regex_drop_cols_X: List=None,
        ) -> None:
        """
        Load data.

        Input:
        - data_dir: path to directory containing files "X_train.csv", "X_test.csv", "Y_train.csv", "Y_test.csv"
        - seed: random seed to use for rng
        - weighted_mse_kwargs: kwargs to use when initializing MSEWeighted. If "weighted_mse" is not selected in trainer, this has no effect.
        """
        self.data_dir = data_dir
        self.seed = seed
        self.weighted_mse_kwargs = {"zeros_mult_factor": 1.} # default values
        if weighted_mse_kwargs is not None:
            self.weighted_mse_kwargs.update(weighted_mse_kwargs)

        self.rng = np.random.default_rng(self.seed)
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        self.data = utils.load_data(
            self.data_dir,
            val=True,
            seed=self.seed,
        )
        self.frac_ones = utils.calc_fraction_ones_per_col(self.data["Y_train"])
        # Define custom loss function, necessary for eval(). Which loss function is actually used is set in train()
        self.weighted_mse_loss_func = MSEWeighted(self.frac_ones, **self.weighted_mse_kwargs).mse_loss
        self.weighted_bce_loss_func = BinaryCrossEntropyWeighted(self.frac_ones).bce_loss

        """cols_to_drop = []
        if regex_drop_cols_X is not None:
            for filtr in regex_drop_cols_X:
                cols_to_drop += list(self.data["X_train"].filter(regex=filtr).columns)
            print(f"\nDropping following columns in X:\n{cols_to_drop}\n")
            self.data["X_train"] = self.data["X_train"].drop(cols_to_drop, axis=1)
            self.data["X_val"] = self.data["X_val"].drop(cols_to_drop, axis=1)
            self.data["X_test"] = self.data["X_test"].drop(cols_to_drop, axis=1)"""

    def train(
        self,
        experiment_name: str="default",
        resample_zeros: bool=False,
        downsample_ones_frac: float=None,
        n_samples: int=None,
        two_stage_training=False,
        fc_model: bool=True, # true to use fc, false to use resnet
        epochs: int=200,
        batch_size: int=128,
        loss_fn: str="weighted_mse",
        lr: float=0.01,
        model_kwargs: dict=None,
        threshold_tn_weight: int=1.,
        ) -> str:
        """
        Train one out of two keras models (fc or resnet). Log to tensorboard, save best model.
        
        Input
        - experiment_name: name of experiment. Used to generate folder where model checkpoint, classification threshold, results and tensorboard statistics are saved
        - resample_zeros: resample data containing 0s
        - downsample_ones_frac: downsample data containing only 1s
        - n_samples: number of samples in X_train to use when training model. Must be <= size of training set
        - two_stage_training: train model on a part of training set, downsample second part and train again
        - fc_model: use fc_model or resnet
        - epochs: training epochs
        - batch_size: batch size
        - loss_fn: loss function: "weighted_mse", "weighted_bce" or function which is compatible with keras model.compile(loss=loss_fn)
        - lr: learning rate
        - model_kwargs: kwargs to use when initializing the fc or resnet model
        - threshold_tn_weight: weight for true negatives (TN) when picking classification threshold. Value > 1 means favor lower TN more than lower TP.

        Output:
        - save_path: path to where 'checkpoint', 'thresholds.py', results and tensorboard statistics are saved
        """
        if fc_model:
            self.model_kwargs = {
                "hidden_dims": [256, 256],
                "use_batch_norm": True,
                "dropout_rate": None, 
                "l2": 0,
                "name": "fc_model"
            }
        else:
            self.model_kwargs = {
                "hidden_dim": 256,
                "n_res_blocks": 5,
                "dropout_rate": None,
                "l2": 0,
                "name": "resnet_model"
            }
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)

        if resample_zeros:
            # Resample data:
            self.data["X_train"], self.data["Y_train"] = pre_process_data_utils.resample_zeros(
                self.data["X_train"], 
                self.data["Y_train"],
                seed=self.seed,
                )
        if downsample_ones_frac:
            assert not resample_zeros, "Cant both resample and downsample, choose one. Set downsample_ones_frac = None to disable"
            self.data["X_train"], self.data["Y_train"] = pre_process_data_utils.downsample_Y_containing_only_ones(
                self.data["X_train"], 
                self.data["Y_train"], 
                desired_fraction_only_ones=downsample_ones_frac,
                rng=self.rng)
        if n_samples:
            self.data["X_train"] = self.data["X_train"][0:n_samples]
            self.data["Y_train"] = self.data["Y_train"][0:n_samples]

        if two_stage_training:
            # Split training data into two parts
            self.data["X_train"], self.data["X_train2"], self.data["Y_train"], self.data["Y_train2"] = train_test_split(
                self.data["X_train"], 
                self.data["Y_train"], 
                test_size=0.3, 
                shuffle=False, 
                #random_state=self.seed
                )
            self.data["X_train2"], self.data["Y_train2"] = pre_process_data_utils.downsample_Y_containing_only_ones(
                self.data["X_train2"], 
                self.data["Y_train2"], 
                desired_fraction_only_ones=0.3,
                rng=self.rng
            )

        logdir = f"experiments/{experiment_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{self.seed}"
        tensorboard_callback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        checkpoint_path = logdir + "/checkpoint"

    
        # Preliminaries for creating model
        input_dim = self.data["X_train"].shape[-1]
        output_dim = self.data["Y_train"].shape[-1] # TODO: handle case where Y is flattened
        norm_layer = layers.Normalization()
        norm_layer.adapt(self.data["X_train"].values) # NOTE: passing in a DataFrame can lead to freeze here

        # Get loss_fn
        if isinstance(loss_fn, str):
            if loss_fn.lower() == "weighted_mse":
                selected_loss_fn = self.weighted_mse_loss_func
            elif loss_fn.lower() == "weighted_bce":
                selected_loss_fn = self.weighted_bce_loss_func
            else:
                raise ValueError(f"Unknown loss function '{loss_fn}'. Choose one of 'weighted_mse' or 'weighted_bc, or pass function directly.'")
        else:
            selected_loss_fn = loss_fn

        # Create model
        if fc_model:
            model = create_fc_model(
                input_dim=input_dim, 
                output_dim=output_dim,
                norm_layer=norm_layer,
                **self.model_kwargs,
                )
        else:
            model = create_resnet_model(
                input_dim=input_dim,
                output_dim=output_dim,
                norm_layer=norm_layer,
                **self.model_kwargs
            )


        model.compile(
            optimizer=optimizers.Adam(lr),
            loss=selected_loss_fn,
            metrics=[
                metrics.RootMeanSquaredError(), 
                metrics.Recall(), # TP rate
                MinRecall(output_dim),
                MaxRecall(output_dim),
                Specificity(), # TN rate
                MinSpecificity(output_dim),
                MaxSpecificity(output_dim),
                #metrics.SensitivityAtSpecificity ,
                #metrics.TrueNegatives(),
                #metrics.AUC(multi_label=True)
                ]
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
        # NOTE: restore_best_weights does not work if early stopping is not triggered! (if model trains for all epochs)
        # Using ModelCheckpoint instead to save best weights

        utils.make_dir_if_doesnt_exist(checkpoint_path)
        # save settings in a config
        config = {
            "data_dir": self.data_dir,
            "seed": self.seed,
            "resample_zeros": resample_zeros,
            "downsample_ones_frac": downsample_ones_frac,
            "n_samples": n_samples,
            "two_stage_training": two_stage_training,
            "fc_model": fc_model,
            "model_kwargs": self.model_kwargs,
            "epochs": epochs,
            "batch_size": batch_size,
            "loss_fn": str(loss_fn),
            "lr": lr,
            "threshold_tn_weight": threshold_tn_weight,
            "features": self.data["X_train"].columns.values,
            "n_features": self.data["X_train"].shape[1],
        }
        if isinstance(loss_fn, str): 
            if loss_fn.lower() == "weighted_mse":
                config["weighted_mse_kwargs"] = self.weighted_mse_kwargs
        utils.save_params(config, logdir + "/config.yml")

        #model.run_eagerly = True

        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        history = model.fit(
            self.data["X_train"].values,
            self.data["Y_train"].values,
            validation_data=(self.data["X_val"].values, self.data["Y_val"].values),
            shuffle=True,
            callbacks=[
                tensorboard_callback, 
                model_checkpoint,
                #early_stopping,
                ],
            epochs=epochs,
            batch_size=batch_size,
        )

        model = self._load_model(checkpoint_path) # load to get best model
        # Select thresholds based on validation set, then save:
        Y_pred_val = model.predict(self.data["X_val"].values)
        selected_thresholds = utils.select_thresholds(self.data["Y_val"], Y_pred_val, tn_weight=threshold_tn_weight, plot=False)
        np.save(logdir + "/thresholds.npy", selected_thresholds)

        if two_stage_training:
            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            history = model.fit(
                self.data["X_train2"].values,
                self.data["Y_train2"].values,
                validation_data=(self.data["X_val"].values, self.data["Y_val"].values),
                shuffle=True,
                callbacks=[
                    tensorboard_callback, 
                    model_checkpoint,
                    #early_stopping,
                    ],
                epochs=epochs,
                batch_size=batch_size,
            )
            model = self._load_model(checkpoint_path) # load to get best model
            # Select thresholds based on validation set, then save:
            Y_pred_val = model.predict(self.data["X_val"].values)
            selected_thresholds = utils.select_thresholds(self.data["Y_val"], Y_pred_val, tn_weight=threshold_tn_weight, plot=False)
            np.save(logdir + "/thresholds.npy", selected_thresholds)
        return logdir


    def _load_model(self, checkpoint_path):
        return tf.keras.models.load_model(
            checkpoint_path, 
            custom_objects={
                "mse_loss": self.weighted_mse_loss_func,
                "bce_loss": self.weighted_bce_loss_func,
                "Specificity": Specificity,
                "MinSpecificity": MinSpecificity,
                "MaxSpecificity": MaxSpecificity,
                "MinRecall": MinRecall,
                "MaxRecall": MaxRecall,
                },
        )


    def eval(
        self, 
        logdir: str, 
        fail_t: bool=False,
        ):
        """
        Evaluate a model.

        logdir: path where 'checkpoint' and optionally 'thresholds.npy' are located.
        fail_t: if model predicts failure timestep instead of binary survival
        """
        checkpoint_path =  os.path.join(logdir, "checkpoint")
        print(f"\nLoading model from checkpoint:\n{checkpoint_path}\n")
        model = self._load_model(checkpoint_path)

        threshold_path = os.path.join(logdir, "thresholds.npy")
        try:
            print(f"\nLoading threshold from:\n{threshold_path}\n")
            thresholds = np.load(threshold_path)
        except OSError:
            print(f"\nCould not find threshold at path:\n{threshold_path}\nSetting thresholds = 0.5\n")
            thresholds = 0.5

        Y_pred_test = model.predict(self.data["X_test"].values)
        Y_pred_classification = (Y_pred_test > thresholds) * 1. 

        if fail_t:
            Y_pred_test_class = pd.DataFrame(Y_pred_classification, columns=self.data["Y_train"].columns)
            Y_pred_test_class = utils.survival_timestep_to_survival(Y_pred_test_class)
            Y_test = utils.survival_timestep_to_survival(self.data["Y_test"])
            tp_col = utils.calc_true_positive_ratio_per_col(Y_test, Y_pred_test_class)
            tn_col = utils.calc_true_negative_ratio_per_col(Y_test, Y_pred_test_class)
            tp = utils.calc_true_positive_ratio(Y_test, Y_pred_test_class)
            tn = utils.calc_true_negative_ratio(Y_test, Y_pred_test_class)
            print(f"Selected thresholds: {thresholds}")
            print(f"True positives per contingency:\n{tp_col}")
            print(f"True negatives per contingency:\n{tn_col}")
            print(f"True positives:\n{tp}")
            print(f"True negatives:\n{tn}")
            print(f"min tp: {tp_col.min()}")
            print(f"min tn: {tn_col.min()}")

        else:
            conf_df_per_col = utils.calc_confusion_metrics_per_col(self.data["Y_test"], Y_pred_classification)
            conf_df = utils.calc_confusion_metrics(self.data["Y_test"], Y_pred_classification)

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
            print(f"Selected thresholds: {thresholds}")
            print(conf_df_per_col)
            print(results_df)
            results_path = os.path.join(logdir, "results.csv")
            if not os.path.isfile(results_path):
                results_df.to_csv(results_path, sep=",")
                print(f"\nSaving results to: {results_path}\n")



def main(seed=99):
    # NOTE: I updated the datasets 20 nov 13.00, so loading and evaluation runs before that will lead to misleading results
    trainer = Trainer(
        data_dir="data/java",
        weighted_mse_kwargs={"zeros_mult_factor": 3.},
        #regex_drop_cols_X=["gen_p_renew"], #["load_p", "gen_p_step", "gen_p_renew"], # Toggle this on and off
        seed=seed,
    )
    cp = trainer.train(
        experiment_name="java_v3",
        epochs=200,
        fc_model=True,
        threshold_tn_weight=3.,
        model_kwargs={
            "hidden_dims": [256, 256],
            "l2": 0,
        },
    )
    #cp = "experiments/java_comp_maintenance_v2_3x0/20231120-142207" # 3x
    #cp = "experiments/java_comp_maintenance_drop_v2/20231123-102727"
    #cp = "experiments/java_comp_maintenance_area_1_drop_v2/20231121-164556"
    #cp = "experiments/java_comp_maintenance_area_1_v2/20231121-162621"
    trainer.eval(cp, fail_t=False)
    # NOTE: prediction y_fail_t benefits disappeared with dynamic thresholds, now significantly worse than normal training process


if __name__ == "__main__":
    main()
    """parser = argparse.ArgumentParser()
    parser.add_argument("seed", help="Seed to use for run", type=int)
    args = parser.parse_args()
    main(int(args.seed))"""
